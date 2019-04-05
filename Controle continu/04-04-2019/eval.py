# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 08:11:08 2019

@author: Clément
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

#TODO : Formater les dates pour l'affichage

def main():
    #Données mesurées, pour l'entrainement
    xTrain = np.array([
            [datetime(2019, 3, 16)],
            [datetime(2019, 3, 18)],
            [datetime(2019, 3, 20)],
            [datetime(2019, 3, 24)],
            [datetime(2019, 3, 25)],
            [datetime(2019, 3, 26)],
            [datetime(2019, 3, 27)],
            [datetime(2019, 3, 28)],
            [datetime(2019, 3, 30)],
            [datetime(2019, 4, 4)]
        ])
        
    yTrain = np.array([[81682], [81720], [81760], [81826], [81844], [81864], [81881], [81900], [81933], [82003]])
    
    #Données qu'on veut prédire
    xTest = np.array([
            [datetime(2019, 4, 5)]
        ])
        
    #Convertir les dates en timestamp
    xTrain = convertir_en_timestamp(xTrain)
    xTest = convertir_en_timestamp(xTest)
    
    #Entrainer modèle
    model = entrainer_modele(xTrain, yTrain)
    
    #Prédire
    yTest = predire(model, xTest)
    
    #Afficher graphique
    afficher_graphique(model, xTrain, yTrain, xTest, yTest)


#Convertir les dates en timestamp
def convertir_en_timestamp(tableau):
    i=0
    for t in tableau:
        tableau[i] = t[0].timestamp()
        i+=1
    return tableau

#Entrainement du modèle
def entrainer_modele(xTrain, yTrain):
    model = LinearRegression()
    model.fit(xTrain, yTrain)
    return model

#Prediction des valeurs
def predire(model, xTest): 
    yTest = model.predict(xTest)
    #Affichage des predictions
    i=0
    for x in xTest:
        print('Le ', datetime.fromtimestamp(x[0]).date(), ' le compteur vaudra ', yTest[i][0])
        i += 1
    return yTest

#Affichage des mesures et du modèle
def afficher_graphique(model, xTrain, yTrain, xTest, yTest):
    plt.scatter(xTrain, yTrain, color='blue', label='Mesure')
    plt.plot(xTrain, model.predict(xTrain), color='red', label='Modèle')
    plt.scatter(xTest, yTest, color='green', label='Prediction')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
