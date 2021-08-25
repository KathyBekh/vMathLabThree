import numpy
from scipy import integrate


def rkf45(function, arguments_T, initialConditions):
    r = integrate.ode(function)\
        .set_integrator('dopri5', atol=0.0001)\
        .set_initial_value(initialConditions, arguments_T[0])
    y = numpy.zeros((len(arguments_T), len(initialConditions)))
    y[0] = initialConditions

    for it in range(1, len(arguments_T)):
        y[it] = r.integrate(arguments_T[it])

        if not r.successful():
            raise RuntimeError('Нельзя!!!')

    return y[:, 0]
