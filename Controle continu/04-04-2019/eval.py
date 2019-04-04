# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 08:11:08 2019

@author: Clément
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

points = np.array([[0,81682.0], [1,81720.0], [2,81760.0], [3,81826.0], [4, 81844.0], [5,81864.0], [6,81881.0], [7,81900.0], [8,81933.0], [9,82003.0]])
teta = np.array([1., 1.])
features = np.array([])
labels = np.array([])
xModel = np.array([])
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