import numpy as np
import pandas as pd
import pylab as P
import matplotlib.pyplot as plt


#Chargement du csv complet
data = pd.read_csv('train.csv', header=0)

'''
print (data.Age.mean())
print(data[['Sex', 'Pclass', 'Age']])
print(data[data['Age'] > 60])
print(data[data['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']])
print(data[data['Age'].isnull()][['Sex', 'Pclass', 'Age']])
'''

#Affichage du nombre d'hommes par classe
'''
for i in range(1,4):
    print(i, len(data[ (data['Sex'] == 'male') & (data['Pclass'] == i) ]))
'''
#Affichage de la repartion des ages

data['Age'].hist(bins=40, range=(0,80))
P.show()

data[data['Survived'] == 0][['Age']].hist(bins=40, range=(0,80), alpha = 0.5)
P.show()
'''
#Ajout d'une colonne avec le genre
data['Gender'] = data['Sex'].map( lambda x: x[0].upper() )
'''
'''dataAge = pd.DataFrame({
        'Survivant' : [data[data['Survived'] == 1]['Age']],
        'Décédé' : [data[data['Survived'] == 0]['Age']]
        })
print(dataAge)'''
#dataAge.plot.hist()
    
#data['Age'].plot.hist(bins=16, alpha = 0.5)
#data[data['Survived'] == 0][['Age']].plot.hist(bins=16, alpha = 0.5)
#data.groupby('Survived')['Age'].plot.hist(bins=16, alpha = 0.5)
    