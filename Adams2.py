import numpy


def adams2(f, t, yO):
    l = len(t) + 2
    y = numpy.zeros(l, len(yO))
    y2 = numpy.zeros(len(t), len(yO))

    h = t[1] - t[0]
    y987 = rk4(f, t[0], t[0] - h, t[0] - 2 * h, yO)
    y[0] = y987[2]
    y[1] = y987[1]
    y[2] = yO

    for it in range(3, len(t) + 2):
        y[it] = y[it - 1] + (h / 2) * (3 * f(t[it - 3] - h, y[it - 1]) - f(t[it - 2], y[it - 2]))

    for it in range(0, len(t)):
        y2[it] = y[it + 2]

    return y2[:, 0]
