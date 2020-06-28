import numpy
import scipy.optimize as optimize
from matplotlib import pylab as plt
from Course1.Week2.Function import week2function


def h(x_val):
    return int(week2function(x_val))


# Строим основную функцию и добавляем её на панель графиков
x = numpy.arange(1, 31, 1)
y = numpy.array([h(i) for i in x])
plt.plot(x, y)

# Поиск локального минимума
minimized = optimize.minimize(h, 30, method="BFGS")
result1 = minimized.x
print(minimized)
print("Solution for x=30 is : " + str(result1))
plt.plot(result1, minimized.fun, "o")

# Поиск глобального минимума
globalMinimum = optimize.differential_evolution(h, [(1, 30)])
print(globalMinimum)
print("Solution for x=30 is : " + str(globalMinimum.x))
plt.plot(globalMinimum.x, globalMinimum.fun, "o")

# Выводим графики на экран
plt.show()
