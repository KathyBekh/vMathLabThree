from math import exp
import numpy
from matplotlib import pyplot
import rkf45
import ExactSolution

# Параметры для решения системы ДУ
a = 1  # Начало отрезка интегрирования
b = 2  # Конец отрезка интегрирования
h = 0.1  # Шаг интегрирования
initialConditions = numpy.array([exp(2), exp(2) * 2])  # Начальные условия


# Исходная система ДУ
def originalFunction(t, y):
    dy = numpy.zeros(y.shape)
    dy[0] = y[1]
    dy[1] = ((t + 1) * y[1] + 2 * (t - 1) * y[0]) / t
    return dy


# вывод результатов на экран
def printResult():
    pyplot.title('Зеленый - ; красный - rkf45, черный - rk4, желтый - rk3, ...')

    t, y_exact = ExactSolution.pick_step(a, b + h, step=h)
    pyplot.plot(t, y_exact, 'g--')

    y_rkf45 = rkf45.rkf45(originalFunction, t, initialConditions)
    pyplot.plot(t, y_rkf45, 'r--')

    # y_rk4 = rk4(originalFunction(), t, initialConditions)
    # pyplot.plot(t, y_rk4, 'k')
    #
    # y_rk3 = rk3(originalFunction(), t, initialConditions)
    # pyplot.plot(t, y_rk3, 'y')
    #
    # y_adams = adams2(f, t, yO)
    # pyplot.plot(t, y_adams, 'm')
    #
    # pyplot.show()

    errer_local_rkf45 = numpy.abs(y_rkf45 - y_exact)
    # errer_local_rk4 = numpy.abs(y_rk4 - y_exact)
    # errer_local_rk3 = numpy.abs(y_rk3 - y_exact)
    # errer_local_adams = numpy.abs(y_adams - y_exact)

    print('First step of RKF45: ', errer_local_rkf45[1])
    # print('First step of RK4: ', errer_local_rk4[1])
    # print('First step of RK3: ', errer_local_rk3[1])
    # print('First step of ADAMS: ', errer_local_adams[1])

    print('Global of RKF45: ', errer_local_rkf45.sum())
    # print('Global of RK4: ', errer_local_rk4.sum())
    # print('Global of RK3: ', errer_local_rk3.sum())
    # print('Global of ADAMS: ', errer_local_adams.sum())

    # print('h^5 is about: ', h ** 5)
    # print('h^4 is about: ', h ** 4)
    # print('h^3 is about: ', h ** 3)
    # print('h^5 / RKF45 first step: ', h ** 5 / errer_local_rkf45[1])
    # # print('h^3 / ADAMS first step: ', h ** 3 / errer_local_adams[1])

    print('\t\t\t\tValues')
    print('t\t\tExact\t\tRKF45\t\tRK4\t\tRK3\t\tADAMS')

    for it in range(0, len(t)):
        print('{:0.1f} \t {}\t {}'.format(t[it],
                                          y_exact[it], y_rkf45[it]))  # , y_rk4[it], y_rk3[it]))  # , y_adams[it]))


if __name__ == '__main__':
    printResult()
