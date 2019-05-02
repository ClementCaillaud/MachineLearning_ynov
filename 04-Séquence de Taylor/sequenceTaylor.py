# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:28:41 2019

@author: Clément
"""

import numpy as np
import matplotlib.pyplot as plt
import math as math

#On a A^x
#On a la dérivée  A^x * ln(A)
#On veut ln(A) = 1
def trouver_A():
    resultat = 0
    a = 5
    nbTours = 0
    
    while(resultat != 1):
        nbTours += 1
        step = 1/nbTours
        resultat = np.log(a)
        
        if(resultat > 1):
            a -= step
        if(resultat < 1):
            a += step
        
    print("---------------------------------------")
    print("On veut trouver A pour avoir  ln(A) = 1")
    print("---------------------------------------")
    print("Nombre d'itérations : ", nbTours)
    print("A = ", a)
    print("ln(A) = ", resultat)


#On a e^x
#On veut approximer e^x avec les séries de Taylor
#e^x = Somme de 0 à n (x^n / n!)
def taylor():
    x = np.arange(-5, 6)
    print("Fontion d'origine : ")
    plt.plot(x, np.exp(x))
    plt.show()
    
    nMax = 8
    for n in range(1, nMax+1):
        print("Approximation n°", n, " :")
        plt.plot(x, np.exp(x))
        plt.plot(x, approximation(x, n))
        plt.show()

def approximation(x, nMax):
    resultat = 0
    for n in range(0, nMax):
        resultat += (x**n) / math.factorial(n)
    return resultat

def main():
    trouver_A()
    taylor()

if __name__ == "__main__":
    main() 

    


    
