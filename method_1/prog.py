import math
import numpy as np


def gauss(sys, n):
    # Метод Гаусса
    sys_copy = np.copy(sys)
    # Прямой ход
    for i in range(0, n):
        if sys_copy[i][i] == 0:
            for j in range(i + 1, n):
                if sys_copy[j][i] != 0:
                    sys_copy[[j, i]] = sys_copy[[i, j]]
                    break
        sys_copy[i] /= sys_copy[i][i]
        for j in range(i + 1, n):
            sys_copy[j] -= sys_copy[i] * sys_copy[j][i]
    # Обратный ход
    x = sys_copy[:, n]
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            x[i] -= sys_copy[i][j] * x[j]
    return x  # Вектор решений


def gauss_main_el(sys, n):
    # Метод Гаусса с выбором главного элемента
    sys_copy = np.copy(sys)
    idx = np.array(range(0, n))
    # Прямой ход
    for i in range(0, n):
        if sys_copy[i][i] == 0:
            for j in range(i + 1, n):
                if sys_copy[j][i] != 0:
                    sys_copy[[j, i]] = sys_copy[[i, j]]
                    break
        # Выбор главного элемента
        max_col_idx = np.argmax(abs(sys_copy[i][i:n])) + i
        idx[i], idx[max_col_idx] = idx[max_col_idx], idx[i]
        sys_copy[:, [i, max_col_idx]] = sys_copy[:, [max_col_idx, i]]
        sys_copy[i] /= sys_copy[i][i]
        for j in range(i + 1, n):
            sys_copy[j] -= sys_copy[i] * sys_copy[j][i]
    # Обратный ход
    x = sys_copy[:, n]
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            x[i] -= sys_copy[i][j] * x[j]
    x_idx = np.array(list(zip(x, idx)))
    # Возврат первоначального порядка неизвестных
    x_idx = x_idx[x_idx[:, 1].argsort()]
    x = x_idx[:, 0]
    return x  # Вектор решений


def det_calc(sys, n):
    # Вычисление определителя
    det = 1
    sys_copy = np.copy(sys[:, :n])
    # Приведение матрицы к верхнему треугольному виду
    for i in range(0, n):
        if sys_copy[i][i] == 0:
            for j in range(i + 1, n):
                if sys_copy[j][i] != 0:
                    sys_copy[[j, i]] = sys_copy[[i, j]]
                    # При перестановке строк знак определителя меняется
                    det *= -1
                    break
            return 0
        det *= sys_copy[i][i]
        for j in range(i + 1, n):
            sys_copy[j] -= sys_copy[i] * sys_copy[j][i] / sys_copy[i][i]
    return det


def inverse(sys, n):
    # Нахождение обратной матрицы
    # Метод Гаусса-Жордана
    # Присоединение единичной матрицы справа
    sys_copy = np.hstack((np.copy(sys[:, :n]), np.identity(n)))
    # Получение верхней треугольной матрицы слева (с единичной диагональю)
    for i in range(0, n):
        if sys_copy[i][i] == 0:
            for j in range(i + 1, n):
                if sys_copy[j][i] != 0:
                    sys_copy[[j, i]] = sys_copy[[i, j]]
                    break
        sys_copy[i] /= sys_copy[i][i]
        for j in range(i + 1, n):
            sys_copy[j] -= sys_copy[i] * sys_copy[j][i]
    # Получение единичной матрицы слева и обратной матрицы справа
    for j in range(n - 1, 0, -1):
        for i in range(j - 1, -1, -1):
            sys_copy[i] -= sys_copy[j] * sys_copy[i][j]
    # Выделение обратной матрицы из расширенной
    inv = sys_copy[:, n:]
    return inv


def matr_norm(sys, n):
    # Вычисляется норма матрицы, согласованная с евклидовой нормой вектора
    norm = 0
    for i in range(0, n):
        for j in range(0, n):
            norm += sys[i][j] ** 2
    return math.sqrt(norm)


def herm_and_pos_def(sys, n, pr_par=1):
    # Проверка матрицы коэффициентов на самосопряжённость и положительную определённость
    herm = 0
    pos_def = 1
    sys_copy = np.copy(sys[:, :n])
    # Самосопряжённость
    if np.array_equal(sys_copy, sys_copy.T):
        herm = 1
    if sys_copy[0][0] > 0:
        for i in range(2, n + 1):
            corn_minor = det_calc(sys_copy[:i, :i], i)
            if corn_minor <= 0:
                pos_def = 0
                break
    if pr_par == 1:
        if (herm == 0) and (pos_def == 0):
            print("\nМатрица коэффициентов данной системы не является самосопряжённой и положительно определённой")
        elif (herm == 0) and (pos_def == 1):
            print("\nМатрица коэффициентов данной системы не является самосопряжённой, но является положительно "
                  "определённой")
        elif (herm == 1) and (pos_def == 0):
            print("\nМатрица коэффициентов данной системы является самосопряжённой, но не является положительно "
                  "определённой")
        else:
            print("\nМатрица коэффициентов данной системы является самосопряжённой и положительно определённой")
    else:
        return herm & pos_def


def relax(sys, n, eps, w):
    # Метод верхней релаксации
    sys_copy = np.copy(sys[:, :n])
    f = np.copy(sys[:, n])
    if herm_and_pos_def(sys_copy, n, 0) == 0:
        # Если матрица не является самосопряжённой и положительно определённой, то она приводится к ней
        f = sys_copy.T @ f
        sys_copy = sys_copy.T @ sys_copy
        cond_num = matr_norm(sys_copy, n) * matr_norm(inverse(sys_copy, n), n)
        print("\nЧисло обусловленности полученной симметрической и положительно определённой матрицы: %.4f" % cond_num)
    else:
        cond_num = matr_norm(sys_copy, n) * matr_norm(inverse(sys_copy, n), n)
        print("\nЧисло обусловленности: %.4f" % cond_num)
    x_new = np.zeros(n)
    discrep_y = eps + 1  # Невязка
    iter = 0
    first_m = np.zeros((n, n))
    for i in range(0, n):
        first_m[i][i] = sys_copy[i][i] / w
    first_m += np.tril(sys_copy, k=-1)
    if det_calc(first_m, n) == 0:
        return 1
    first_m = inverse(first_m, n)
    while (eps <= discrep_y) and (iter <= 500000):
        iter += 1
        x_prev = np.copy(x_new)
        second_m = f - (sys_copy @ x_prev)
        x_new = (first_m @ second_m) + x_prev
        discrep_y = (np.linalg.norm(f - (sys_copy @ x_new), 2)) * np.linalg.norm(np.linalg.inv(sys_copy), 2)

    if iter > 500000:
        print("\nКол-во итераций превысило 500000")
    else:
        print("\nРешения, полученные с помощью метода верхней релаксации:")
        for i in range(0, n):
            print("x_%d = %.7f" % (i + 1, x_new[i]))
        print("\nКол-во итераций:", iter)


def formula(n, m):
    # Получение матрицы по специальным формулам (приложение 2, п. 1-2)
    sys = np.zeros((n, n + 1))
    for i in range(1, n + 1):
        sys[i - 1][n] = 200 + 50 * i
        for j in range(1, n + 1):
            if i == j:
                sys[i - 1][j - 1] = n + m * m + j / m + i / n
            else:
                sys[i - 1][j - 1] = (i + j) / (m + n)
    return sys


def run():
    # Основная программа
    global matrix
    choice = 1
    while choice == 1:
        print("\nКаким методом Вы хотите решать систему?\n1 - методы Гаусса (1 подзадание)\n2 - метод верхней "
              "релаксации (2 подзадание)")
        task_num = int(input("\nВведите соответствующий номер: "))
        if (task_num < 1) or (task_num > 2):
            print("\nНеправильный номер")
            continue
        print("\nКак Вы хотите ввести систему: 1 - из файла, 2 - с помощью формулы?")
        mode = int(input("\nВведите соответствующий номер: "))
        if mode == 1:
            sys_num = int(input("\nНомер системы, которую Вы хотите ввести (от 1 до 3 - приложение 1-9, от 4 до 6 - "
                                "приложение 1-3): "))
            if (sys_num > 6) or (sys_num < 1):
                print("\nНеправильный номер системы\n")
                continue
            else:
                file_name = str("sys" + str(sys_num) + ".txt")
                f = open(file_name, 'r')
                file_contents = f.read()
                print("\nВы выбрали систему\n", file_contents, sep='')
                matrix = np.loadtxt(file_name)
        elif mode == 2:
            print()
            n = 20
            m = 8
            matrix = formula(n, m)
            print("\nПолученная система:")
            for i in range(0, n):
                for j in range(0, n):
                    print("%*.4f " % (7, matrix[i][j]), end="")
                print("| %*.4f " % (7, matrix[i][n]))

        else:
            print("\nНеправильный способ")
            continue

        cnt = matrix.shape[0]
        det = det_calc(matrix, cnt)
        print("\nОпределитель: %.7f" % det)
        if det == 0:
            print("\nДанная система вырождена, поэтому имеет бесконечно много решений")
        elif task_num == 1:
            inv = inverse(matrix, cnt)
            print("\nОбратная матрица:")
            for i in range(0, cnt):
                for j in range(0, cnt):
                    print("%*.4f " % (7, inv[i][j]), end="")
                print()

            cond_num = matr_norm(matrix, cnt) * matr_norm(inv, cnt)
            print("\nЧисло обусловленности: %.4f" % cond_num)

            x_gauss = gauss(matrix, cnt)
            print("\nРешения, полученные с помощью метода Гаусса:")
            for i in range(0, cnt):
                print("x_%d = %.7f" % (i + 1, x_gauss[i]))

            x_gauss_m_el = gauss_main_el(matrix, cnt)
            print("\nРешения, полученные с помощью метода Гаусса с выбором главного элемента:")
            for i in range(0, cnt):
                print("x_%d = %.7f" % (i + 1, x_gauss_m_el[i]))
        else:
            choice_2 = 1
            herm_and_pos_def(matrix, cnt)
            cond_num = matr_norm(matrix, cnt) * matr_norm(inverse(matrix, cnt), cnt)
            print("\nЧисло обусловленности: %.4f" % cond_num)
            while choice_2 == 1:
                try:
                    eps = float(input("\nВведите положительную точность eps, с которой Вы хотите получить решения: "))
                    if eps <= 0:
                        raise RuntimeError
                except:
                    eps = -1
                    while eps <= 0:
                        try:
                            eps = float(input("\nВведена неправильная точность. Попробуйте ещё раз: "))
                        except:
                            eps = -1
                try:
                    w = float(input("\nВведите положительный итерационный параметр w\n (для симметрической "
                                    "положительно определенной матрицы системы следует выбирать 0 < w < 2): "))
                    if w <= 0:
                        raise RuntimeError
                except:
                    w = -1
                    while w <= 0:
                        try:
                            w = float(input("\nВведён неправильный параметр. Попробуйте ещё раз: "))
                        except:
                            w = -1
                relax(matrix, cnt, eps, w)

                try:
                    choice_2 = int(input("\n\nВведите 1, если хотите заново протестировать метод при других "
                                         "параметрах, 0 - если хотите закончить тестирование: "))
                    if (choice_2 != 0) and (choice_2 != 1):
                        raise RuntimeError
                except:
                    choice_2 = -1
                    while (choice_2 != 0) and (choice_2 != 1):
                        try:
                            choice_2 = int(input("\nВведён неправильный номер. Попробуйте ещё раз: "))
                        except:
                            choice_2 = -1
            print("\n\nТестирование метода верхней релаксации закончено")
        try:
            choice = int(input("\n\nВведите 1, если хотите заново запустить программу, 0 - если хотите выйти: "))
            if (choice != 0) and (choice != 1):
                raise RuntimeError
        except:
            choice = -1
            while (choice != 0) and (choice != 1):
                try:
                    choice = int(input("\nВведён неправильный номер. Попробуйте ещё раз: "))
                except:
                    choice = -1

    print("\nВыход")


run()  # Запуск программы
