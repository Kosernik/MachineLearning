import numpy as np
import pandas as pd
import seaborn as sns
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Задание 1

# Чтение данных из файла
data = pd.read_csv('data\weights_heights.csv', index_col='Index')

# График распределения роста (в дюймах)
# data.plot(y='Height', kind='hist', color='red',  title='Height (inch.) distribution')

# Первые 5 значений дата-сета
print(data.head(5))

# График распределения веса (в фунтах)
# data.plot(y='Weight', kind='hist', color='green',  title='Weight (pounds.) distribution')


def make_bmi(height_inch, weight_pound):
    '''
    Метод возвращает BMI - индекс отношения массы к росту
    :param height_inch: высота в дюймах
    :param weight_pound: вес в фунтах
    :return: значение индекса
    '''
    METER_TO_INCH, KILO_TO_POUND = 39.37, 2.20462
    return (weight_pound / KILO_TO_POUND) / \
           (height_inch / METER_TO_INCH) ** 2

# Добавление в дата-сет BMI
data['BMI'] = data.apply(lambda row: make_bmi(row['Height'], row['Weight']), axis=1)
print(data.head(5))

# Строим таблицу графиков отношений массы, роста, BMI
# sns.pairplot(data)


def weight_category(weight):
    '''
    Метод распределения по весу в одну из 3-х категорий
    :param weight: вес в фунтах
    :return: группа, в которую входит данный вес
    '''
    if weight < 120:
        return 1
    elif weight >= 150:
        return 3
    else:
        return 2

# Добавление в дата-сет категории весов
data['weight_cat'] = data['Weight'].apply(weight_category)
print(data.head(5))

# Бокс-плот распределения весовых категорий по росту
# sns.boxplot(x='weight_cat', y='Height', data=data)

# График отношения роста и веса
data.plot(x='Weight', y='Height', kind='scatter', color='blue',  title='Зависимость роста от веса')


# Задание 2


def minSquareError(w):
    '''
    Вычисление квадрата ошибок
    :param w0: свободный коэффициент
    :param w1: коэффициент при весе
    :param data: дата-сет, содержащий рост и вес
    :return: сумма квадратов ошибок
    '''
    error = 0
    for i in range(1, len(data)):
        error += (data['Height'][i] - (w[0] + w[1] * data["Weight"][i])) ** 2
    return error / len(data)


def fiftySquare(w1):
    return minSquareError((50.0, w1))


def line(w0, w1, x):
    return w0 + w1*x


weights = np.linspace(data['Weight'].min(), data['Weight'].max())
# y1 = line(60, 0.05, weights)
# y2 = line(50, 0.16, weights)
# plt.plot(weights, y1, 'r', label='First')
# plt.plot(weights, y2, 'y', label='Second')


# w1 = np.linspace(-5, 5)
# yError = [minSquareError((i, 50.0)) for i in w1]
# plt.plot(w1, yError)

# [9]. С помощью метода minimize_scalar из scipy.optimize найдите минимум функции, определенной в п. 6, для значений
# параметра  w1w1  в диапазоне [-5,5]. Проведите на графике из п. 5 Задания 1 прямую, соответствующую значениям
# параметров ( w0w0 ,  w1w1 ) = (50,  w1_optw1_opt ), где  w1_optw1_opt  – найденное в п. 8 оптимальное значение
# параметра  w1w1 .
# w1_opt = opt.minimize_scalar(fiftySquare, bounds=[-5, 5])
# print(w1_opt)
#
# yAxis = line(50, w1_opt.x, weights)
# plt.plot(weights, yAxis, 'g', label='Err')

# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# X = np.arange(-100, 100, 0.5)
# Y = np.arange(-10, 10, 0.05)
# X, Y = np.meshgrid(X, Y)
# Z = minSquareError((X, Y))
#
# surf = ax.plot_surface(X, Y, Z)
# ax.set_xlabel('Intercept')
# ax.set_ylabel('Slope')
# ax.set_zlabel('Error')

sol = opt.minimize(minSquareError, np.array([0.0, 0.0]), bounds=((-100, 100), (-5, 5)), method='L-BFGS-B')
print("Solution:")
print(sol)

solutionLine = line(sol.x[0], sol.x[1], weights)
plt.plot(weights, solutionLine, 'r', label='Solution')

# Выводим графики на экран
plt.show()
