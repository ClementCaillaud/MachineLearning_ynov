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

#Données mesurées, pour l'entrainement
#16/03
#18/03
#20/03
#24/03
#25/03
#26/03
#27/03
#28/03
#30/03
#04/04
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
i=0
for d in xTrain:
    xTrain[i] = d[0].timestamp()
    i+=1
    
yTrain = np.array([[81682], [81720], [81760], [81826], [81844], [81864], [81881], [81900], [81933], [82003]])

#Données qu'on veut prédire
#05/04
xTest = np.array([
        [datetime(2019, 4, 5)]
    ])
i=0
for d in xTest:
    xTest[i] = d[0].timestamp()
    i+=1

#Entrainement du modèle
model = LinearRegression()
model.fit(xTrain, yTrain)

#Prediction des valeurs
yTest = model.predict(xTest)

#Affichage des mesures et du modèle
plt.scatter(xTrain, yTrain, color='blue', label='Mesure')
plt.plot(xTrain, model.predict(xTrain), color='red', label='Modèle')
plt.scatter(xTest, yTest, color='green', label='Prediction')
plt.legend()
plt.show()

#Affichage des predictions
i=0
for x in xTest:
    print('Le ', datetime.fromtimestamp(x[0]).date(), ' le compteur vaudra ', yTest[i][0])
    i += 1
