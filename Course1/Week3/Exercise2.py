import numpy
import scipy.optimize as optimize
from matplotlib import pylab as plt
from Course1.Week2.Function import week2function


# Строим основную функцию и добавляем её на панель графиков
x = numpy.arange(1, 30, 0.1)
y = week2function(x)
plt.plot(x, y)

# Находим глобальный минимум на отрезке функции X от 1 до 30
globalMinimum = optimize.differential_evolution(week2function, [(1, 30)])
print(globalMinimum.x)
print(globalMinimum.fun)
plt.plot(globalMinimum.x, globalMinimum.fun, "o")

# Выводим графики на экран
plt.show()
