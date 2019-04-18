# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:30:40 2019

@author: Clément CAILLAUD
@author: Antoine DROUARD
"""

"""
On souhaite utiliser un jeu de données réel avec :
* une random forest 
* une régression logistique OvR
* une régression logistique OvO
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from timeit import default_timer as timer

def main():
    #Chargement des données
    faces = fetch_olivetti_faces()
    #Séparation data / target
    data = faces['data']
    target = faces['target']
    #Création d'un jeu de train et de test
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    #Quelques infos sur les données
    print("400 photographies du visage de 40 personnes")
    print("Entrainement sur 80% des données")
    print("Test sur 20% des données")
    print()
    
    random_forest(x_train, x_test, y_train, y_test)

def ovo(x_train, x_test, y_train, y_test):
    print("Régression logistique OvO")
    

def ovr(x_train, x_test, y_train, y_test):
    print("Régression logistique OvR")
    
def random_forest(x_train, x_test, y_train, y_test):
    print("Random forest")
    #Création du classifieur
    rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    #Entrainement
    time_start = timer()
    rfc.fit(x_train, y_train)
    time_end = timer()
    print("L'entrainement a duré ", time_end - time_start, " secondes")
    #Prédiction sur le jeu de test
    prediction(rfc, x_test, y_test)

def prediction(classifieur, x_test, y_test):
    #Initialisation statistiques
    nb_predictions = 0
    nb_predictions_correctes = 0
    #Tests de prédicition sur le jeu de test
    time_start = timer()
    for key, x in enumerate(x_test):
        if classifieur.predict([x]) == y_test[key]:
            nb_predictions_correctes += 1
        nb_predictions += 1
    time_end = timer()
    print("La prédiction a duré ", time_end - time_start, " secondes")
    #Affichage précision
    print("Précision de la prédiction : ", (nb_predictions_correctes / nb_predictions) * 100, "%")

if __name__ == "__main__":
    main()   