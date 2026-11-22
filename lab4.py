"""
Лабораторная работа №8. Решение уравнения Лапласа методом сеток.

Вариант 14.   u_xx + u_yy = 0  в области D,
    граница Г:  (|x| + 2)(|y| + 2) = 12   ("скруглённый ромб",
                 минимальный описанный прямоугольник: -4 <= x <= 4, -4 <= y <= 4)
    u|Г = 2|x| + 0.5|y|

Криволинейная граница НЕ проходит через узлы сетки, поэтому (по указанию
методички "определить значения u в граничных узлах с возможно меньшей
погрешностью"):
  - граничными считаются внутренние узлы, у которых есть соседний узел
    вне области (т.е. узлы, ближайшие к границе);
  - значение в граничном узле берётся из граничного условия в ближайшей
    точке кривой (проекция узла на границу).

Сетка:  x_m = -4 + m*h,  y_n = -4 + n*h,   h1 = h2 = h = 1/K
(методичка: шаги h1, h2 порядка 0.1, т.е. K = 10).

Разностное уравнение (8.3) при h1 = h2:
    u[m,n] = (u[m-1,n] + u[m+1,n] + u[m,n-1] + u[m,n+1]) / 4
Система решается итерационно:
    1 - метод Зейделя,  2 - метод наискорейшего спуска.

Запуск:
    python lab4.py                 - параметры спросит программа
    python lab4.py <метод> <K>     - например:  python lab4.py 1 10
        K = 1/h:  K=10 -> h=0.1 (как в методичке),  K=5 -> h=0.2 (быстрее)
"""
import math
import sys
import time
import numpy as np

# ------------------------------------------------------------------ область
# D: (|x| + 2)(|y| + 2) <= 12;  описанный прямоугольник [-4, 4] x [-4, 4]
XMIN, XMAX = -4, 4


def region_value(x, y):
    """< 0 внутри области, = 0 на границе, > 0 снаружи."""
    return (np.abs(x) + 2.0) * (np.abs(y) + 2.0) - 12.0


def boundary_value(x, y):
    return 2.0 * np.abs(x) + 0.5 * np.abs(y)


def curve_r(theta):
    """Полярный радиус точки границы в направлении theta:
    (r|cos| + 2)(r|sin| + 2) = 12  =>  r = 16/(2(a+b) + sqrt(4(a+b)^2 + 32ab)),
    a = |cos theta|, b = |sin theta| (устойчивая форма через сопряжённое)."""
    a, b = np.abs(np.cos(theta)), np.abs(np.sin(theta))
    return 16.0 / (2.0 * (a + b) + np.sqrt(4.0 * (a + b) ** 2 + 32.0 * a * b))


def curve_points(n=8192):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = curve_r(th)
    return th, r * np.cos(th), r * np.sin(th)


def project_on_boundary(px, py, th, cx, cy):
    """Ближайшая точка кривой Г к точке (px, py):
    грубый минимум по плотной выборке + уточнение золотым сечением."""
    k0 = int(np.argmin((cx - px) ** 2 + (cy - py) ** 2))
    lo, hi = th[k0] - 2.0 * th[1], th[k0] + 2.0 * th[1]

    def dist2(t):
        r = float(curve_r(t))
        return (r * math.cos(t) - px) ** 2 + (r * math.sin(t) - py) ** 2

    gr = (math.sqrt(5.0) - 1.0) / 2.0           # золотое сечение
    c1 = hi - gr * (hi - lo)
    d1 = lo + gr * (hi - lo)
    fc, fd = dist2(c1), dist2(d1)
    for _ in range(80):
        if hi - lo < 1e-14:
            break
        if fc < fd:
            hi, d1, fd = d1, c1, fc
            c1 = hi - gr * (hi - lo)
            fc = dist2(c1)
        else:
            lo, c1, fc = c1, d1, fd
            d1 = lo + gr * (hi - lo)
            fd = dist2(d1)
    t = 0.5 * (lo + hi)
    r = float(curve_r(t))
    return r * math.cos(t), r * math.sin(t)


def build_grid(K):
    """Сетка с граничными узлами: in-region узлы, имеющие соседа вне области."""
    nx = ny = 8 * K + 1
    xs = np.array([-4.0 + i / K for i in range(nx)])
    ys = np.array([-4.0 + j / K for j in range(ny)])
    X, Y = xs[:, None], ys[None, :]
    inreg = region_value(X, Y) <= 1e-9           # внутри области или на границе
    outside = ~inreg

    # граничные узлы: внутри области, но есть сосед (по 4 направлениям) снаружи
    near_out = np.zeros((nx, ny), dtype=bool)
    near_out[1:-1, :] |= outside[:-2, :] | outside[2:, :]
    near_out[:, 1:-1] |= outside[:, :-2] | outside[:, 2:]
    bound = inreg & near_out
    inside = inreg & ~bound

    u = np.zeros((nx, ny))
    th, cx, cy = curve_points()
    for i, j in zip(*np.nonzero(bound)):
        xb, yb = project_on_boundary(xs[i], ys[j], th, cx, cy)
        u[i, j] = boundary_value(xb, yb)
    u[inside] = u[bound].mean()                  # начальное приближение
    return xs, ys, inside, bound, u


# ------------------------------------------------------------------ методы
def seidel(u0, inside, K, eps, max_iter):
    """Метод Зейделя: новые значения используются сразу."""
    u = u0.tolist()
    nx = len(u)
    # для каждого столбца i - список внутренних узлов j (граница криволинейная,
    # поэтому внутри столбца возможны и граничные узлы - берём точные индексы)
    columns = {}
    for i in range(1, nx - 1):
        js = np.nonzero(inside[i])[0]
        if len(js):
            columns[i] = [int(j) for j in js]
    hist = []
    for it in range(1, max_iter + 1):
        dmax = 0.0
        for i, js in columns.items():
            col, um, up = u[i], u[i - 1], u[i + 1]
            for j in js:
                new = 0.25 * (um[j] + up[j] + col[j - 1] + col[j + 1])
                d = abs(new - col[j])
                if d > dmax:
                    dmax = d
                col[j] = new
        hist.append(dmax)
        if dmax < eps:
            break
    return np.array(u), it, hist


def steepest_descent(u0, inside, eps, max_iter):
    """Наискорейший спуск для A u = b, A = -Лаплас (симметрична, положительно определена)."""
    u = u0.copy()
    mask = inside.astype(float)

    def neighbors(a):
        s = np.zeros_like(a)
        s[1:-1, :] += a[:-2, :] + a[2:, :]
        s[:, 1:-1] += a[:, :-2] + a[:, 2:]
        return s

    hist = []
    for it in range(1, max_iter + 1):
        r = (neighbors(u) - 4.0 * u) * mask          # r = b - A u
        Ar = (4.0 * r - neighbors(r)) * mask
        rr = np.sum(r * r)
        if rr == 0.0:
            break
        alpha = rr / np.sum(r * Ar)
        u += alpha * r
        rmax = np.max(np.abs(alpha * r))
        hist.append(rmax)
        if rmax < eps:
            break
    return u, it, hist


def direct_reference(u, inside):
    """Прямое решение той же разностной системы (для проверки итераций)."""
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
    except ImportError:
        return None
    idx = -np.ones(inside.shape, dtype=int)
    pts = list(zip(*np.nonzero(inside)))
    for k, (i, j) in enumerate(pts):
        idx[i, j] = k
    A = sp.lil_matrix((len(pts), len(pts)))
    b = np.zeros(len(pts))
    for k, (i, j) in enumerate(pts):
        A[k, k] = 4.0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if inside[ii, jj]:
                A[k, idx[ii, jj]] = -1.0
            else:
                b[k] += u[ii, jj]
    sol = spl.spsolve(A.tocsr(), b)
    ref = u.copy()
    for k, (i, j) in enumerate(pts):
        ref[i, j] = sol[k]
    return ref


# ------------------------------------------------------------------ вывод
def print_table(xs, ys, u, inside, bound, K):
    """Таблица значений в узлах с шагом 1 по x и по y (внутри области)."""
    cols = list(range(0, len(xs), K))
    print("\nПриближённое решение u(x, y) (шаг таблицы: по x и по y - 1;  '.' - вне области)")
    print(f"{'y \\ x':>7}" + "".join(f"{xs[i]:>8.1f}" for i in cols))
    for j in range(len(ys) - 1, -1, -K):
        row = f"{ys[j]:>7.1f}"
        for i in cols:
            row += f"{u[i, j]:>8.3f}" if (inside[i, j]
                                          or bound[i, j]) else f"{'.':>8}"
        print(row)


def plot(xs, ys, u, inside, bound, hist, name, K):
    import matplotlib.pyplot as plt

    region = inside | bound
    Z = np.ma.masked_where(~region, u).T            # строки - y, столбцы - x
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
        13, 6.5), gridspec_kw={"width_ratios": [1, 1.4]})

    cf = ax1.contourf(xs, ys, Z, levels=20, cmap="viridis")
    ax1.contour(xs, ys, Z, levels=20, colors="k", linewidths=0.3)
    th, cx, cy = curve_points()
    ax1.plot(cx, cy, "r-", lw=1.5, label="граница Г")
    fig.colorbar(cf, ax=ax1, shrink=0.8, label="u")
    ax1.set_aspect("equal")
    ax1.set_title(f"Решение u(x, y), h = {1.0 / K:g}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.semilogy(range(1, len(hist) + 1), hist, "b-")
    ax2.set_title(f"Сходимость: {name}")
    ax2.set_xlabel("номер итерации")
    ax2.set_ylabel("max |u_new - u_old|")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    plt.show()


def ask_params():
    if len(sys.argv) >= 3:
        return sys.argv[1], int(sys.argv[2])
    print("Итерационный метод:\n  1 - метод Зейделя\n  2 - метод наискорейшего спуска")
    m = ""
    while m not in ("1", "2"):
        m = input("Выберите метод: ").strip()
    k = input("K = 1/h (10 -> h=0.1, 5 -> h=0.2) [10]: ").strip()
    return m, int(k) if k else 10


def main():
    method, K = ask_params()
    h = 1.0 / K
    xs, ys, inside, bound, u0 = build_grid(K)
    n_in, n_b = int(inside.sum()), int(bound.sum())

    print(f"\nВариант 14: u_xx + u_yy = 0, область (|x|+2)(|y|+2) <= 12, u|Г = 2|x| + 0.5|y|")
    print(f"Шаги h1 = h2 = {h:g};  узлов: внутренних {n_in}, граничных {n_b}")

    eps = 1e-8
    t = time.time()
    if method == "1":
        name = "метод Зейделя"
        u, it, hist = seidel(u0, inside, K, eps, 200000)
    else:
        name = "метод наискорейшего спуска"
        u, it, hist = steepest_descent(u0, inside, eps, 200000)
    print(f"{name}: {it} итераций, {time.time() - t:.1f} с, критерий max|u_new - u_old| < {eps:g}")

    print_table(xs, ys, u, inside, bound, K)

    # проверки
    umin, umax = u[inside].min(), u[inside].max()
    bmin, bmax = u[bound].min(), u[bound].max()
    print(f"\nПринцип максимума: u внутри [{umin:.4f}, {umax:.4f}] в пределах граничных [{bmin:.4f}, {bmax:.4f}]"
          f" -> {'выполнен' if bmin - 1e-9 <= umin and umax <= bmax + 1e-9 else 'НАРУШЕН'}")
    res = max(abs(u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j])
              for i, j in zip(*np.nonzero(inside)))
    print(f"Невязка разностных уравнений: max|сумма соседей - 4u| = {res:.2e}")
    ref = direct_reference(u0, inside)
    if ref is not None:
        print(
            f"Отличие от прямого решения той же системы: {np.max(np.abs(u - ref)):.2e}")

    plot(xs, ys, u, inside, bound, hist, name, K)


if __name__ == "__main__":
    main()
