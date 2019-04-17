# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:11:40 2019

@author: Clément
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import datasets
from sklearn.model_selection import train_test_split

def main():
    
    #Chargement des données
    digits = datasets.load_digits()
    
    #Séparation données et cibles
    data = digits['data']
    target = digits['target']
    
    #Création d'un jeu d'entrainement et d'un jeu de test
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
    
    #Création du classifieur
    rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    
    #Entrainement
    rfc.fit(x_train, y_train)
    
    #Prédiction sur le jeu de test
    predict(rfc, x_test, y_test)
    
    
def predict(rfc, x_test, y_test):
    
    #Initialisation statistiques
    nb_predictions = 0
    nb_predictions_correctes = 0
    
    #Tests de prédicition sur le jeu de test
    for key, x in enumerate(x_test):
        if rfc.predict([x]) == y_test[key]:
            nb_predictions_correctes += 1
        nb_predictions += 1
    
    #Affichage précision
    print("Précision de la prédiction : ", (nb_predictions_correctes / nb_predictions) * 100, "%")
    
if __name__ == "__main__":
    main()    