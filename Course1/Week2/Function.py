import numpy


def week2function(x_val):
    return numpy.sin(x_val / 5.0) * numpy.exp(x_val / 10.0) + 5 * numpy.exp(-x_val / 2.0)