import numpy
import scipy.optimize as optimize
from matplotlib import pylab as plt
from Course1.Week2.Function import week2function


# Строим основную функцию и добавляем её на панель графиков
x1 = 2.0
x2 = 30.0
x = numpy.arange(1, 30, 0.1)
# print(x)
y = week2function(x)
# print(y)
plt.plot(x, y)


# Ищем локальный минимум для начального приближения x=2
minimized = optimize.minimize(week2function, x1, method="BFGS")
print(minimized)
result1 = minimized.x
print("Solution for x1=" + str(x1) + " : " + str(result1))
# Добавляем точку на график
plt.plot(result1, week2function(result1), "o")

# Ищем локальный минимум для начального приближения x=2
minimized = optimize.minimize(week2function, x2, method="BFGS")
print(minimized)
result2 = minimized.x
print("Solution for x1=" + str(x2) + " : " + str(result2))
# Добавляем точку на график
plt.plot(result2, week2function(result2), "o")

# Выводим графики на экран
plt.show()
