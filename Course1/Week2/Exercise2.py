import pandas
import numpy
import scipy.linalg as sl
from matplotlib import pylab as plt


def f(x_val):
    return numpy.sin(x_val / 5.0) * numpy.exp(x_val / 10.0) + 5 * numpy.exp(-x_val / 2.0)


def g1(x_val, b):
    return b[0] + b[1] * x_val


x = numpy.arange(1, 15, 0.1)
y = f(x)
plt.plot(x, y)

print(f(1))
print(f(15))

A1 = numpy.array([[1.0, 1.0], [1.0, 15.0]])
b1 = numpy.array([f(1), f(15)])
s1 = sl.solve(A1, b1)
print(s1)
y1 = g1(x, s1)
# plt.plot([1.0, 15.0], [g1(1.0, s1), g1(15.0, s1)])
plt.plot(x, y1)

A2 = numpy.array([[1.0, 1.0, 1.0],
                  [1.0, 8.0, 8.0**2],
                  [1.0, 15.0, 15.0**2]])
b2 = numpy.array([f(1), f(8), f(15)])
s2 = sl.solve(A2, b2)
print(s2)
def g2(x_val, b):
    return b[0] + b[1] * x_val + b[2] * (x_val**2)

y2 = g2(x, s2)
plt.plot(x, y2)
# plt.plot([1.0, 8.0, 15.0], [g2(1.0, s2), g2(8.0, s2), g2(15.0, s2)])

A3 = numpy.array([[1.0, 1.0, 1.0, 1.0],
                  [1.0, 4.0, 4.0**2, 4.0**3],
                  [1.0, 10.0, 100.0, 1000.0],
                  [1.0, 15.0, 15.0**2, 15.0**3]])
b3 = numpy.array([f(1.0), f(4.0), f(10.0), f(15.0)])
s3 = sl.solve(A3, b3)
print(s3)
def g3(x_val, b):
    return b[0] + b[1] * x_val + b[2] * (x_val ** 2) + b[3] * (x_val ** 3)

y3 = g3(x, s3)
plt.plot(x, y3)

print(f(15))
print(g3(15.0, s3))

plt.show()
