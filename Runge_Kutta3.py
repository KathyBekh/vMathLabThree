import numpy


def rk3(function, arguments_T, initialConditions):
    y = numpy.zeros((len(arguments_T), len(initialConditions)))
    y[0] = initialConditions
    h = arguments_T[1] - arguments_T[0]

    for it in range(1, len(arguments_T)):
        k1 = function(arguments_T[it - 1], y[it - 1])
        k2 = function(arguments_T[it - 1] + 0.5 * h, y[it - 1] + 0.5 * h * k1)
        k3 = function(arguments_T[it - 1] + 3.0 * h / 4, y[it - 1] + 3.0 * h * k2 / 4)
        y[it] = y[it - 1] + h * (2 * k1 + 3 * k2 + 4 * k3) / 9.0
    return y


def rk3Values(function, arguments_T, initialConditions):
    y = rk3(function, arguments_T, initialConditions)
    return y[:, 0]
