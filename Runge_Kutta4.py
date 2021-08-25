import numpy


def rk4(f, t, yO):
    y = numpy.zeros((len(t), len(yO)))
    y[0] = yO
    h = t[1] - t[0]

    for it in range(1, len(t)):
        k1 = f(t[it - 1], y[it - 1])
        k2 = f(t[it - 1] + 0.5 * h, y[it - 1] + 0.5 * h * k1)
        k3 = f(t[it - 1] + 0.5 * h, y[it - 1] + 0.5 * h * k2)
        k4 = f(t[it - 1] + h, y[it - 1] + h * k3)
        y[it] = y[it - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return y[:, 0]