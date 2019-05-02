# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:35:32 2019

@author: Clément
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import random
import cv2
import numpy as np

def main():
    #Chargement des données de train et de test, une image est de dimension 8*8 = 64 pixels
    x_train, x_test, y_train, y_test = chargement_donnees()
    #Création d'un réseau de neurones
    reseau = creer_reseau(64, 10)
    #Entrainement du réseau
    nb_entrainements = 4
    for i in range(0, nb_entrainements):
        reseau = entrainer_reseau(reseau, x_train, y_train, 1)

    dessiner_reseau(reseau)
    #Prédictions sur les valeurs de test
    predictions = predire(reseau, x_test)
    statistiques_prediction(predictions, y_test)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def chargement_donnees():
    """Retourne les données d'entrainement et de test"""
    digits = datasets.load_digits()
    data = digits['data']
    target = digits['target']
    return train_test_split(data, target, test_size=0.3)

def creer_reseau(nb_entrees, nb_sorties):
    """Créé un réseau de neurones avec nb_entrees entrées et nb_sorties sorties"""
    tab_neurones = []
    for i in range(0, nb_sorties):
        tab_neurones.append(creer_neurone(64))
    return tab_neurones
    
def creer_neurone(nb_entrees):
    """Créé un neuronne réagissant à nb_entrees entrées"""
    tab_poids = [random.randint(50, 201) for i in range(0, nb_entrees)]
    return tab_poids

def entrainer_reseau(reseau, x_train, y_train, coefficient):
    """Entraine le réseau de neurones avec les données de train"""
    for key_image, image in enumerate(x_train):
        #On regarde ce que prédit notre model
        prediction = predire(reseau, [image])[0]
        #Si la prédiction n'est pas bonne, on augmente le poids des pixels allumés et on réduit les autres pour la classe attendue
        if prediction != y_train[key_image]:
            reseau[y_train[key_image]] = ajuster_poids(reseau[y_train[key_image]], image, coefficient)
    return reseau

def ajuster_poids(neurone, image, coefficient):
    """Ajuste les poids du neurone"""
    for key_pixel, pixel in enumerate(image):
        if pixel > 0 and neurone[key_pixel] < 255:
            neurone[key_pixel] += coefficient
        elif neurone[key_pixel] > 0:
            neurone[key_pixel] -= coefficient
    return neurone

def predire(reseau, x_test):
    """Cherche à prédire les nombres présents sur les images"""
    predictions = []
    #On procède pour chaque image
    for key_image, image in enumerate(x_test):
        tab_total = [0 for i in reseau]
        #On va tester chaque classe et regarder laquelle ressort le total le plus élevé
        for key_neurone, neurone in enumerate(reseau):
            tab_total[key_neurone] = valeur_sortie_neurone(neurone, image)
        #On récupère la classe qui a le plus grand résultat
        valeur_max = max(tab_total)
        prediction = tab_total.index(valeur_max)
        predictions.append(prediction)
    return predictions
            
def valeur_sortie_neurone(neurone, image):
    """Retourne la sortie du neurone"""
    total = 0
    #On multiplie l'état de chaque entrée par le poid associé et on l'additionne au total
    for key_pixel, pixel in enumerate(image):
        if(pixel > 0):
            total += neurone[key_pixel]
    return total

def statistiques_prediction(predictions, y_test):
    """Affiche la précision des prédictions"""
    nb_predictions = len(predictions)
    nb_predictions_correctes = 0
    for key_prediction, prediction in enumerate(predictions):
        if prediction == y_test[key_prediction]:
            nb_predictions_correctes += 1
    print("Nombre d'images testées :", nb_predictions)
    print("Nombre de prédictions correctes :", nb_predictions_correctes)
    print("Précision :", round((nb_predictions_correctes * 100) / nb_predictions, 2), "%")

def dessiner_reseau(reseau):
    """Dessine l'état des poids du réseau"""
    vertical_1 = np.vstack((np.uint8(reseau[0]).reshape(8, 8), np.uint8(reseau[5]).reshape(8, 8)))
    vertical_2 = np.vstack((np.uint8(reseau[1]).reshape(8, 8), np.uint8(reseau[6]).reshape(8, 8)))
    vertical_3 = np.vstack((np.uint8(reseau[2]).reshape(8, 8), np.uint8(reseau[7]).reshape(8, 8)))
    vertical_4 = np.vstack((np.uint8(reseau[3]).reshape(8, 8), np.uint8(reseau[8]).reshape(8, 8)))
    vertical_5 = np.vstack((np.uint8(reseau[4]).reshape(8, 8), np.uint8(reseau[9]).reshape(8, 8)))
    img = img = np.hstack((vertical_1, vertical_2, vertical_3, vertical_4, vertical_5))
    img = cv2.resize(np.uint8(img), (960, 384), interpolation = cv2.INTER_AREA)
    cv2.imshow("img", img)

if __name__ == "__main__":
    main()   