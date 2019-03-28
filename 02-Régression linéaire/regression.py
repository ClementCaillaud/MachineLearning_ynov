# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:43:19 2019

@author: Clément
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

points = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
teta = np.array([1., 1.])
features = np.array([])
labels = np.array([])
xModel = np.array([1, 1, 2, 2])
yModel = np.array([])
m = points.size / 2
alpha = 2

def h(x):
    return teta[0] + teta[1] * x

def J():
    somme = 0
    for point in points:
        somme += (h(point[0]) - point[1]) ** 2
    return (1/(2*m)) * somme

def new_Teta0():
    somme = 0
    for point in points:
        somme += h(point[0]) - point[1]
    return teta[0] - (alpha / m) * somme

def new_Teta1():
    somme = 0
    for point in points:
        somme += (h(point[0]) - point[1]) * point[0]
    return teta[1] - (alpha / m) * somme

#Régression linéaire
valeurJ = 0
for n in range(1, 101):
    #print(teta)
    #print(J())
    alpha = 10/n
    tempTeta0 = new_Teta0()
    tempTeta1 = new_Teta1()
    teta[0] = tempTeta0
    teta[1] = tempTeta1

print(teta)
print(J())

#Séparation des X et des Y
for point in points:
    features = np.append(features, point[0])
    labels = np.append(labels, point[1])

#Calcul des valeurs theoriques
for x in xModel:
    yModel = np.append(yModel, h(x))

#Affichage points et model
plt.scatter(features, labels, color="blue")
plt.plot(xModel, yModel, color="red")
axes = plt.axes()
axes.grid()
plt.show()

reg = LinearRegression().fit(points, yModel)
print(reg.score(points, yModel))
print(reg.coef_)
print(reg.intercept_)
print(reg.predict(np.array([[1.5,5]])))
