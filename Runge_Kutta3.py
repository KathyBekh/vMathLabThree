import numpy


def rk3(f, t, yO):
    y = numpy.zeros((len(t), len(yO)))
    y[0] = yO
    h = t[1] - t[0]

    for it in range(1, len(t)):
        k1 = f(t[it - 1], y[it - 1])
        k2 = f(t[it - 1] + 0.5 * h, y[it - 1] + 0.5 * h * k1)
        k3 = f(t[it - 1] + 3.0 * h / 4, y[it - 1] + 3.0 * h * k2 / 4)
        y[it] = y[it - 1] + (2 * k1 + 3 * k2 + 4 * k3) / 9.0
    return y[:, 0]