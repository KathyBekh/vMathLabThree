import numpy


def rk4(function, arguments_T, initialConditions):
    y = numpy.zeros((len(arguments_T), len(initialConditions)))
    y[0] = initialConditions
    h = arguments_T[1] - arguments_T[0]

    for it in range(1, len(arguments_T)):
        k1 = function(arguments_T[it - 1], y[it - 1])
        k2 = function(arguments_T[it - 1] + 0.5 * h, y[it - 1] + 0.5 * h * k1)
        k3 = function(arguments_T[it - 1] + 0.5 * h, y[it - 1] + 0.5 * h * k2)
        k4 = function(arguments_T[it - 1] + h, y[it - 1] + h * k3)
        y[it] = y[it - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return y


def rk4Values(function, arguments_T, initialConditions):
    y = rk4(function, arguments_T, initialConditions)
    return y[:, 0]
