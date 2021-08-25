import math
import numpy


# Точное решение системы уравнений
def exact(t):
    y = numpy.zeros(len(t))
    for it in range(0, len(t)):
        y[it] = math.exp(2.0 * t[it])
    return y


# Функция pick_step возвращает массив точек t и точных значений в этих точках y(t)
def pick_step(a, b, step):
    t = numpy.arange(a, b, step)
    y = exact(t)
    return t, y
