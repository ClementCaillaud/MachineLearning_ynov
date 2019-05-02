# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:43:19 2019

@author: Clément
"""

"""
L'objectif est de tracer le séparateur de la régression linéaire sans utiliser sklearn, à la main
"""

import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression

def h(x, teta):
    return teta[0] + teta[1] * x

def J(points, m, teta):
    somme = 0
    for point in points:
        somme += (h(point[0], teta) - point[1]) ** 2
    return (1/(2*m)) * somme

def new_Teta0(points, teta, alpha, m):
    somme = 0
    for point in points:
        somme += h(point[0], teta) - point[1]
    return teta[0] - (alpha / m) * somme

def new_Teta1(points, teta, alpha, m):
    somme = 0
    for point in points:
        somme += (h(point[0], teta) - point[1]) * point[0]
    return teta[1] - (alpha / m) * somme

#Régression linéaire
def main():
    points = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    teta = np.array([1., 1.])
    features = np.array([])
    labels = np.array([])
    xModel = np.array([1, 1, 2, 2])
    yModel = np.array([])
    m = points.size / 2
    alpha = 2
    
    print("On place", len(points), "points :", points)
    print("On cherche à trouver la valeur de Teta_0 et Teta_1 pour tracer le séparateur linéaire")
    
    for n in range(1, 101):
        alpha = 10/n
        tempTeta0 = new_Teta0(points, teta, alpha, m)
        tempTeta1 = new_Teta1(points, teta, alpha, m)
        teta[0] = tempTeta0
        teta[1] = tempTeta1
    
    print("Teta :", teta)
    print("J :", J(points, m, teta))
    
    #Séparation des X et des Y
    for point in points:
        features = np.append(features, point[0])
        labels = np.append(labels, point[1])
    
    #Calcul des valeurs theoriques
    for x in xModel:
        yModel = np.append(yModel, h(x, teta))
    
    #Affichage points et model
    plt.scatter(features, labels, color="blue", label="points")
    plt.plot(xModel, yModel, color="red", label="separateur")
    axes = plt.axes()
    axes.grid()
    print("Séparateur linéaire :")
    plt.show()
    
    """reg = LinearRegression().fit(points, yModel)
    print(reg.score(points, yModel))
    print(reg.coef_)
    print(reg.intercept_)
    print(reg.predict(np.array([[1.5,5]])))"""

if __name__ == "__main__":
    main() 