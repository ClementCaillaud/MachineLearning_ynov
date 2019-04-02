# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:07:51 2019

@author: Clément
"""

import numpy as np
import matplotlib.pyplot as plt
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset  UserID::MovieID::Rating::Timestamp
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.15)

def user_collaborative_filtering():
    
    # Use user_based true/false to switch between user-based or item-based collaborative filtering
    algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
    algo.fit(trainset)
    
    # we can now query for specific predicions
    uid = str(196)  # raw user id
    iid = str(302)  # raw item id
    
    # get a prediction for specific users and items.
    pred = algo.predict(uid, iid, r_ui=4, verbose=True)
    
    # run the trained model against the testset
    test_pred = algo.test(testset)
    
    # get RMSE
    print("User-based Model : Test Set")
    accuracy.rmse(test_pred, verbose=True)
    
    # if you wanted to evaluate on the trainset
    #print("User-based Model : Training Set")
    #train_pred = algo.test(trainset.build_testset())
    #accuracy.rmse(train_pred)

user_collaborative_filtering()