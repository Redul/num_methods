import numpy as np
import math
import matplotlib.pyplot as plt


# y' = 3 - x - y, y(0) = 0
def f1(x, y):
    return 3 - x - y


# Точное решение теста выше
# y = 4 - x - 4*exp(-x)
def sol1(x):
    y = list()
    for i in x:
        y.append(4 - i - 4 * math.exp(-i))
    return y


# y' = (y - e^x)^2 + e^x, y(0) = 0
def f2(x, y):
    return (y - math.exp(x)) ** 2 + math.exp(x)


# Точное решение теста выше
# y = exp(x) - 1 /(x + 1)
def sol2(x):
    y = list()
    for i in x:
        y.append(math.exp(i) - 1 / (i + 1))
    return y


# Первая система (решения, представимого в элементарных функциях, не существует):
# (y_1)' = sin(2 * y_1^2) + y_2 + x, y_1(0) = 1
# (y_2)' = x + y_1 - 2 * (y_2^2) + 1, y_2(0) = 0.5
def f1_1(x, y):
    return math.sin(2 * (y[0] ** 2)) + x + y[1]


def f1_2(x, y):
    return x + y[0] - 2 * (y[1] ** 2) + 1


# Вторая система
# (y_1)' = -y_2, y_1(0) = 1
# (y_2)' = y_1, y_2(0) = 0
def f2_1(x, y):
    return -y[1]


def f2_2(x, y):
    return y[0]


# Решения:
# y_1 = cos(x)
# y_2 = sin(x)
def sol2_1(x):
    y = list()
    for i in x:
        y.append(math.cos(i))
    return y


def sol2_2(x):
    y = list()
    for i in x:
        y.append(math.sin(i))
    return y


def R_K_2(func, x_0, y_0, func_cnt, n, len):
    # Метод Рунге-Кутта 2-ого порядка
    # используется схема "предиктор - корректор"
    h = len / n  # шаг
    if func_cnt == 1:
        # Одно уравнение
        grid = dict()
        grid[x_0] = y_0
        x_i = x_0
        y_i = y_0
        for i in range(0, n):
            x_new = x_i + h
            grid[x_new] = y_i + (func(x_i, y_i) + func(x_new, y_i + h * func(x_i, y_i))) * h / 2
            y_i = grid[x_new]
            x_i = x_new
        return grid
    else:
        # Система уравнений
        grid = dict()
        grid[x_0] = y_0
        x_i = x_0
        y_i = y_0
        for i in range(0, n):
            x_new = x_i + h
            y_vals1 = np.zeros(func_cnt)
            y_vals2 = np.zeros(func_cnt)
            for j in range(0, func_cnt):
                y_vals1[j] = func[j](x_i, y_i)
                y_vals2[j] = func[j](x_new, y_i + h * y_vals1[j])
            grid[x_new] = y_i + (y_vals1 + y_vals2) * h / 2
            y_i = grid[x_new]
            x_i = x_new
        return grid


def R_K_4(func, x_0, y_0, func_cnt, n, len):
    # Метод Рунге-Кутта 4-ого порядка
    h = len / n  # шаг
    grid = dict()
    grid[x_0] = y_0
    x_i = x_0
    y_i = y_0
    if func_cnt == 1:
        # Одно уравнение
        for i in range(0, n):
            x_new = x_i + h
            k1 = func(x_i, y_i)
            k2 = func(x_i + h / 2, y_i + (h / 2) * k1)
            k3 = func(x_i + h / 2, y_i + (h / 2) * k2)
            k4 = func(x_new, y_i + h * k3)
            grid[x_new] = y_i + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            y_i = grid[x_new]
            x_i = x_new
        return grid
    else:
        # Система уравнений
        for i in range(0, n):
            x_new = x_i + h
            k1 = np.zeros(func_cnt)
            k2 = np.zeros(func_cnt)
            k3 = np.zeros(func_cnt)
            k4 = np.zeros(func_cnt)
            for j in range(0, func_cnt):
                k1[j] = func[j](x_i, y_i)
            for j in range(0, func_cnt):
                k2[j] = func[j](x_i + h / 2, y_i + (h / 2) * k1)
            for j in range(0, func_cnt):
                k3[j] = func[j](x_i + h / 2, y_i + (h / 2) * k2)
            for j in range(0, func_cnt):
                k4[j] = func[j](x_new, y_i + h * k3)
            grid[x_new] = y_i + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            y_i = grid[x_new]
            x_i = x_new
        return grid


def run():
    cnt = int(input(
        "Введите 1, если хотите протестировать решение дифференциального уравнения\nВведите 2, если хотите "
        "протестировать решение системы из двух дифференциальных уравнений\n"))
    global f, x_0, y_0, sol
    if cnt == 1:
        num = int(input("Введите номер уравнения, решение которого вы хотите получить:\n"
                        "1: y' = 3 - x - y, y(0) = 0\n"
                        "2: y' = (y - e^x)^2 + e^x, y(0) = 0\n"))
        if num == 1:
            f = f1
            sol = sol1
            x_0 = 0
            y_0 = 0
        else:
            f = f2
            sol = sol2
            x_0 = 0
            y_0 = 0
        len = float(input("Введите длину отрезка, на котором вы хотите получить решение:\n"))
        n = int(input("Введите кол-во узлов (точек), который будут составлять решение:\n"))
        print("\nШаг: %.4f\n" % (len / n))
        grid_2 = R_K_2(f, x_0, y_0, cnt, n, len)
        grid_4 = R_K_4(f, x_0, y_0, cnt, n, len)

        # ------------------------------------------
        # Ниже - ввод и вывод данных и построение графиков
        # ------------------------------------------

        print("   x\t\ty (R_K_2)\ty (R_K_4)\t   y (Точное решение)\n")
        sols = list(sol(grid_2.keys()))
        for i in range(0, n):
            x = list(grid_2.keys())[i]
            print("%.6f\t%.6f\t%.6f\t\t%.6f\n" % (x, list(grid_2.values())[i], list(grid_4.values())[i], sols[i]))
        print()
        plt.subplot(121)
        x = np.linspace(x_0, x_0 + len, 5000)
        plt.plot(x, sol(x), c='red', lw=3, label='Точное решение')
        plt.plot(list(grid_2.keys()), list(grid_2.values()), '--', c='blue', lw=3, label='y (R_K_2)')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        plt.subplot(122)
        plt.plot(x, sol(x), c='red', lw=3, label='Точное решение')
        plt.plot(list(grid_4.keys()), list(grid_4.values()), '--', c='blue', lw=3, label='y (R_K_4)')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

    else:
        num = int(input("Введите номер системы уравнений, решение которой вы хотите получить:\n"
                        "1: { (y_1)' = sin(2 * y_1^2) + y_2 + x, y_1(0) = 1,\n"
                        "     (y_2)' = x + y_1 - 2 * (y_2^2) + 1, y_2(0) = 0.5 }\n"
                        "2: { (y_1)' = -y_2, y_1(0) = 1,\n"
                        "     (y_2)' = y_1, y_2(0) = 0 }\n"))
        if num == 1:
            f = [f1_1, f1_2]
            x_0 = 0
            y_0 = [1, 0.5]
        else:
            f = [f2_1, f2_2]
            sol = [sol2_1, sol2_2]
            x_0 = 0
            y_0 = [1, 0]
        len = float(input("Введите длину отрезка, на котором вы хотите получить решения:\n"))
        n = int(input("Введите кол-во узлов (точек), который будут составлять решения:\n"))
        print("\nШаг: %.4f\n" % (len / n))
        grid_2 = R_K_2(f, x_0, y_0, cnt, n, len)
        grid_4 = R_K_4(f, x_0, y_0, cnt, n, len)
        vals_2_1 = list()
        for i in grid_2.values():
            vals_2_1.append(i[0])
        vals_2_2 = list()
        for i in grid_2.values():
            vals_2_2.append(i[1])
        vals_4_1 = list()
        for i in grid_4.values():
            vals_4_1.append(i[0])
        vals_4_2 = list()
        for i in grid_4.values():
            vals_4_2.append(i[1])
        if num == 2:
            print("   x\t\ty_1 (R_K_2)\ty_1 (R_K_4)\t   y (Точное решение)\n")
            sols = list(sol[0](grid_2.keys()))
            for i in range(0, n):
                x = list(grid_2.keys())[i]
                print("%.6f\t%.6f\t%.6f\t\t%.6f\n" % (x, vals_2_1[i], vals_4_1[i], sols[i]))
            print()
            plt.subplot(221)
            x = np.linspace(x_0, x_0 + len, 5000)
            plt.plot(x, sol[0](x), c='red', lw=3, label='Точное решение')
            plt.plot(list(grid_2.keys()), vals_2_1, '--', c='blue', lw=3, label='y_1 (R_K_2)')
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(loc=0)

            plt.subplot(222)
            plt.plot(x, sol[0](x), c='red', lw=3, label='Точное решение')
            plt.plot(list(grid_4.keys()), vals_4_1, '--', c='blue', lw=3, label='y_1 (R_K_4)')
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(loc=0)

            print("\n   x\t\ty_2 (R_K_2)\ty_2 (R_K_4)\t   y (Точное решение)\n")
            sols = list(sol[1](grid_2.keys()))
            for i in range(0, n):
                x = list(grid_2.keys())[i]
                print("%.6f\t%.6f\t%.6f\t\t%.6f\n" % (x, vals_2_2[i], vals_4_2[i], sols[i]))
            print()
            plt.subplot(223)
            x = np.linspace(x_0, x_0 + len, 5000)
            plt.plot(x, sol[1](x), c='red', lw=3, label='Точное решение')
            plt.plot(list(grid_2.keys()), vals_2_2, '--', c='blue', lw=3, label='y_2 (R_K_2)')
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(loc=0)

            plt.subplot(224)
            plt.plot(x, sol[1](x), c='red', lw=3, label='Точное решение')
            plt.plot(list(grid_4.keys()), vals_4_2, '--', c='blue', lw=3, label='y_2 (R_K_4)')
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(loc=0)
        else:
            print("   x\t\ty_1 (R_K_2)\ty_1 (R_K_4)\n")
            for i in range(0, n):
                x = list(grid_2.keys())[i]
                print("%.6f\t%.6f\t%.6f\n" % (x, vals_2_1[i], vals_4_1[i]))
            print()
            plt.subplot(221)
            plt.plot(list(grid_2.keys()), vals_2_1, c='blue', lw=3, label='y_1 (R_K_2)')
            grid_2 = R_K_2(f, x_0, y_0, cnt, n // 2, len)
            vals_2_1 = list()
            for i in grid_2.values():
                vals_2_1.append(i[0])
            plt.plot(list(grid_2.keys()), vals_2_1, c='green', lw=3)
            grid_2 = R_K_2(f, x_0, y_0, cnt, n * 2, len)
            vals_2_1 = list()
            for i in grid_2.values():
                vals_2_1.append(i[0])
            plt.plot(list(grid_2.keys()), vals_2_1, c='red', lw=3)
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(loc=0)

            plt.subplot(222)
            plt.plot(list(grid_4.keys()), vals_4_1, c='blue', lw=3, label='y_1 (R_K_4)')
            grid_4 = R_K_4(f, x_0, y_0, cnt, n // 2, len)
            vals_4_1 = list()
            for i in grid_4.values():
                vals_4_1.append(i[0])
            plt.plot(list(grid_4.keys()), vals_4_1, c='green', lw=3)
            grid_4 = R_K_2(f, x_0, y_0, cnt, n * 2, len)
            vals_4_1 = list()
            for i in grid_4.values():
                vals_4_1.append(i[0])
            plt.plot(list(grid_4.keys()), vals_4_1, c='red', lw=3)
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(loc=0)

            grid_2 = R_K_2(f, x_0, y_0, cnt, n, len)
            grid_4 = R_K_4(f, x_0, y_0, cnt, n, len)
            print("\n   x\t\ty_2 (R_K_2)\ty_2 (R_K_4)\n")
            for i in range(0, n):
                x = list(grid_2.keys())[i]
                print("%.6f\t%.6f\t%.6f\n" % (x, vals_2_2[i], vals_4_2[i]))
            print()
            plt.subplot(223)
            plt.plot(list(grid_2.keys()), vals_2_2, c='blue', lw=3, label='y_2 (R_K_2)')
            grid_2 = R_K_2(f, x_0, y_0, cnt, n // 2, len)
            vals_2_2 = list()
            for i in grid_2.values():
                vals_2_2.append(i[1])
            plt.plot(list(grid_2.keys()), vals_2_2, c='green', lw=3)
            grid_2 = R_K_2(f, x_0, y_0, cnt, n * 2, len)
            vals_2_2 = list()
            for i in grid_2.values():
                vals_2_2.append(i[1])
            plt.plot(list(grid_2.keys()), vals_2_2, c='red', lw=3)
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(loc=0)

            plt.subplot(224)
            plt.plot(list(grid_4.keys()), vals_4_2, c='blue', lw=3, label='y_2 (R_K_4)')
            grid_4 = R_K_4(f, x_0, y_0, cnt, n // 2, len)
            vals_4_2 = list()
            for i in grid_4.values():
                vals_4_2.append(i[1])
            plt.plot(list(grid_4.keys()), vals_4_2, c='green', lw=3)
            grid_4 = R_K_2(f, x_0, y_0, cnt, n * 2, len)
            vals_4_2 = list()
            for i in grid_4.values():
                vals_4_2.append(i[1])
            plt.plot(list(grid_4.keys()), vals_4_2, c='red', lw=3)
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(loc=0)


run()  # Запуск программы
