# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:28:41 2019

@author: Clément
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression # import de la classe

#chargement de base de données iris
iris = datasets.load_iris()

# choix de deux variables
X = iris.data[:, :2] # Utiliser les deux premiers colonnes afin d'avoir un problème de classification binaire.&nbsp;
y = (iris.target != 0) * 1 # re-étiquetage des fleurs

#visualisation des données
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='g', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='y', label='1')
plt.legend();
plt.show()

model = LogisticRegression(solver='lbfgs') # construction d'un objet de Régression logistique
model.fit(X, y) # Entrainement du modèle

Iries_To_Predict = [
    [5.5, 2.5],
    [7, 3],
    [3,2],
    [5,3]
]
predicted = model.predict(Iries_To_Predict)
print(predicted)
