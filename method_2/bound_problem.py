import numpy as np
import math
import matplotlib.pyplot as plt

global f, p, q, a_1, a_2, a, c_1, b_1, b_2, b, c_2, x_g, y_g


# p = p(x) - функция при первой производной
# q = q(x) - функция при y
# f - правая часть уравнения
# a_1, b_1 - коэффициенты в 1 дополнительном условии
# a_2, b_2 - коэффициенты в 2 дополнительном условии
# a, b - левая и правая границы
# c_1, c_2 - правая часть доп. условий
# x_g - сетка
# y_g - сеточная функция

# y'' - 0.5xy' + y = 2, y(0.4) = 1.2, y(0.7) + 2y'(0.7) = 1.4
# Решения, представимого в элементарных функциях, не существует
def p_1(x):
    return -0.5 * x


def q_1(x):
    return 1


def f_1(x):
    return 2


def probl_1():
    global f, p, q, a_1, a_2, a, c_1, b_1, b_2, b, c_2
    f = f_1
    p = p_1
    q = q_1
    a_1 = 1
    a_2 = 0
    a = 0.4
    c_1 = 1.2
    b_1 = 1
    b_2 = 2
    b = 0.7
    c_2 = 1.4


# y'' + y' = 1, y'(0) = 0, y(1) = 1
# Решение: y = x + exp(-x) - 1/exp(1)
def sol_2(x):
    y = list()
    for i in x:
        y.append(i + math.exp(-i) - 1 / math.exp(1))
    return y


def p_2(x):
    return 1


def q_2(x):
    return 0


def f_2(x):
    return 1


def probl_2():
    global f, p, q, a_1, a_2, a, c_1, b_1, b_2, b, c_2
    f = f_2
    p = p_2
    q = q_2
    a_1 = 0
    a_2 = 1
    a = 0
    c_1 = 0
    b_1 = 1
    b_2 = 0
    b = 1
    c_2 = 1


# y'' - 6y/x^2 = 0, y(0) + y'(0) = 0, y(1) = 2
# Решение: y = 2x^3
def sol_3(x):
    y = list()
    for i in x:
        y.append(2 * (i ** 3))
    return y


def p_3(x):
    return 0


def q_3(x):
    return -6 / (x ** 2)


def f_3(x):
    return 0


def probl_3():
    global f, p, q, a_1, a_2, a, c_1, b_1, b_2, b, c_2
    f = f_3
    p = p_3
    q = q_3
    a_1 = 1
    a_2 = 1
    a = 0
    c_1 = 0
    b_1 = 1
    b_2 = 0
    b = 1
    c_2 = 2


def solution_task(n):
    # Решение задачи с помощью метода прогонки
    global x_g, y_g
    h = (b - a) / n
    x_i = a
    x_g = np.zeros(n + 1)
    x_g[0] = x_i
    y_g = np.zeros(n + 1)
    # Прогоночные коэффициенты
    alpha = np.zeros(n + 1)
    beta = np.zeros(n + 1)
    alpha[1] = -a_2 / (a_1 * h - a_2)
    beta[1] = c_1 * h / (a_1 * h - a_2)
    for i in range(1, n):
        x_i += h
        x_g[i] = x_i
        # Коэффициенты трёхдиагональной СЛАУ
        A_i = 1 / (h * h) - p(x_i) / (2 * h)
        C_i = 2 / (h * h) - q(x_i)
        B_i = 1 / (h * h) + p(x_i) / (2 * h)
        D_i = f(x_i)
        # Вычисление прогоночных коэффициентов по реккурентным формулам
        alpha[i + 1] = B_i / (C_i - A_i * alpha[i])
        beta[i + 1] = (beta[i] * A_i - D_i) / (C_i - A_i * alpha[i])
    # Обратная прогонка
    y_g[n] = (b_2 * beta[n] + c_2 * h) / (b_2 * (1 - alpha[n]) + b_1 * h)
    for i in range(n, 0, -1):
        y_g[i - 1] = y_g[i] * alpha[i] + beta[i]
    x_g[n] = b


def run():
    # Основная программа
    num = int(input("Введите номер краевой задачи, для которой вы хотите найти решение:\n"
                    "1: y'' - 0.5xy' + y = 2, y(0.4) = 1.2, y(0.7) + 2y'(0.7) = 1.4\n"
                    "2: y'' + y' = 1, y'(0) = 0, y(1) = 1\n"
                    "3: y'' - 6y/x^2 = 0, y(0) + y'(0) = 0, y(1) = 6\n"))
    probl = [probl_1, probl_2, probl_3]
    probl[num - 1]()
    sol = [0, sol_2, sol_3]
    n = int(input("\nВведите желаемое кол-во итераций: "))
    solution_task(n)
    if num == 1:
        print("\n   x\t\t   y\n")
        for i in range(0, n + 1):
            print("%.6f\t%.6f\n" % (x_g[i], y_g[i]))
        print()
        plt.plot(x_g, y_g, c='blue', lw=3, label="y")
        solution_task(n // 2)
        plt.plot(x_g, y_g, c='green', lw=3, label="y (в 2 раза больше итераций)")
        solution_task(n * 2)
        plt.plot(x_g, y_g, c='red', lw=3, label="y (в 2 раза меньше итераций)")
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc=0)

    else:
        vals = sol[num - 1](x_g)
        print("\n   x\t\t   y\t   y (Точное решение)\n")
        for i in range(0, n + 1):
            print("%.6f\t%.6f\t\t%.6f\n" % (x_g[i], y_g[i], vals[i]))
        o_x = np.linspace(a, b, 5000)
        plt.plot(o_x, sol[num - 1](o_x), c='red', lw=3, label="Точное решение")
        plt.plot(x_g, y_g, '--', c='blue', lw=3, label="y")
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc=0)


run()  # Запуск программы
