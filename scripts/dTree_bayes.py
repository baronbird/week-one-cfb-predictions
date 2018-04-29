#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 16:02:46 2018

@author: sammyalptekin
"""

import matplotlib.pyplot as plt
from graphviz import Source
from sklearn import svm, metrics, grid_search, tree
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.datasets import load_iris
from pandas import DataFrame
import numpy as np
import heapq


# Initial Data
training = DataFrame.from_csv("../data/raw_data_normalized_t.csv")
testing = DataFrame.from_csv("../data/raw_data_normalized_test.csv")

# Data - NA's
training = training.dropna(axis=1, how='all')
testing = testing.dropna(axis=1, how='all')
training = training.dropna(axis=0, how='any')
testing = testing.dropna(axis=0, how='any')

# Just the Labels
trainLabel = training['Label']
testLabel = testing['Label']

# Home and Away Teams
trainHomes = training['home_team']
trainAways = training['away_team']
testHomes = testing['home_team']
testAways = testing['away_team']

# Drop Home, Away, and Label from the data
training = training.drop('away_team', 1).drop('home_team', 1).drop("Label", 1)
#training = training.drop("", 1)
testing = testing.drop('away_team', 1).drop('home_team', 1).drop("Label", 1)
#testing = testing.drop("year", 1)

iris = load_iris()


machine = tree.DecisionTreeRegressor( criterion="mse" )
machine.fit(training.as_matrix(), trainLabel.as_matrix())
anew = metrics.accuracy_score(np.array(testLabel), machine.predict(testing.as_matrix()).round())
print ("ID3      : ", anew)

# Create a Decision Tree by iteratively adding the best features (independant of each other) to the model
def feature_selection_best_ind(training, testing):
    # Keep a priority in order to rank the features
    pq = []
    
    # Rank the features in order of F1 score when operating alone
    for feature in training.columns:
        # Select one feature only
        train = training[feature]
        test = testing[feature]
        
        # Determine its F1 score
        machine = tree.DecisionTreeRegressor( criterion="mse" )
        machine.fit(train.as_matrix().reshape(-1,1), trainLabel.as_matrix() )
        F1 = metrics.f1_score(np.array(testLabel), machine.predict(test.as_matrix().reshape(-1,1)).round())
        
        
        heapq.heappush(pq, (-F1, feature))
    [print(thing) for thing in pq]
    print(heapq.heappop(pq))
    print(heapq.heappop(pq))

def createDTree(training, testing):
    feature_selection_best_ind(training, testing)
    feature_selection_best_dep(training, testing)
    feature_selection_worst(training, testing)

#machine = KNeighborsClassifier(p=len(training.columns.values)*2)
#machine = tree.DecisionTreeClassifier( criterion="entropy" )

feature_selection_best_ind(training, testing)

print(type(machine))
print("Prediction:", machine.predict( testing.as_matrix() ).round())
print("Real Value:", np.array(testLabel).round())

print("Confusion Matrix:\n", metrics.confusion_matrix( np.array(testLabel), machine.predict( testing.as_matrix() ).round()))
print("Accuracy:", metrics.accuracy_score( np.array(testLabel), machine.predict( testing.as_matrix() ).round()))
print("F1 Score:", metrics.f1_score( np.array(testLabel), machine.predict( testing.as_matrix() ).round()))
print("Precision:", metrics.precision_score( np.array(testLabel), machine.predict( testing.as_matrix() ).round()))
print("Recall:", metrics.recall_score( np.array(testLabel), machine.predict( testing.as_matrix() ).round()))

'''
graph = Source( tree.export_graphviz(machine, out_file=None, feature_names=training.columns))
png_bytes = graph.pipe(format='png')
with open('dtree_pipe_proposed_model.png','wb') as f:
    f.write(png_bytes)

from IPython.display import Image
Image(png_bytes) 
'''