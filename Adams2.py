import numpy
import Runge_Kutta3


def adams2(function, arguments_T, initialConditions):
    l = len(arguments_T) + 2
    y = numpy.zeros((l, len(initialConditions)))
    y2 = numpy.zeros((len(arguments_T), len(initialConditions)))

    h = arguments_T[1] - arguments_T[0]
    startPoints = Runge_Kutta3.rk3(function, [arguments_T[0], arguments_T[0] - h, arguments_T[0] - 2 * h],
                                   initialConditions)
    y[0] = startPoints[2]
    y[1] = startPoints[1]
    y[2] = initialConditions

    for it in range(2, l-1):
        y[it] = y[it - 1] + (h / 2) * (3 * function(arguments_T[it - 1], y[it - 1]) - function(arguments_T[it - 2], y[it - 2]))

    for it in range(0, len(arguments_T)):
        y2[it] = y[it + 2]

    return y2[:, 0]
