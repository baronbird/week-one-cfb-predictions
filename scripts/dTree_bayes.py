#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 16:02:46 2018

@author: sammyalptekin
"""

import matplotlib.pyplot as plt
from graphviz import Source
from sklearn import svm, metrics, grid_search, tree, naive_bayes
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

'''
machine = tree.DecisionTreeRegressor( criterion="mse" )
machine.fit(training.as_matrix(), trainLabel.as_matrix())
anew = metrics.accuracy_score(np.array(testLabel), machine.predict(testing.as_matrix()).round())
print ("ID3      : ", anew)
'''

# Create a Decision Tree by iteratively adding the best features (independant of each other) to the model
def feature_selection_best_ind(train_d, test_d):
    # Keep a priority queue in order to rank the features
    pq = []
    
    # Rank the features in order of F1 score when operating alone
    for feature in train_d.columns:
        # Select one feature only
        train = train_d[feature]
        test = test_d[feature]
        
        # Determine its F1 score
        machine = tree.DecisionTreeRegressor( criterion="mse" )
        machine.fit(train.as_matrix().reshape(-1,1), trainLabel.as_matrix() )
        F1 = metrics.f1_score(np.array(testLabel), machine.predict(test.as_matrix().reshape(-1,1)).round())

        # Add it to the priority queue
        heapq.heappush(pq, (-F1, feature))
    
    
    # While adding a feature improves the F1 score, keep adding features
    old = 0
    new = .01
    features = []
    x = []
    y = []
    #while new > old:
    while pq:
        # Consider another feature
        features.append(heapq.heappop(pq)[1])
        train = train_d[features]
        test = test_d[features]
        
        # Uncomment a machine
        #machine = tree.DecisionTreeRegressor( criterion="mse" )
        machine = naive_bayes.GaussianNB()
        #machine = KNeighborsClassifier(p=len(training.columns.values)*2)
        #machine = tree.DecisionTreeClassifier( criterion="entropy" )
        
        # Determine the F1 score of the model
        machine.fit(train.as_matrix(), trainLabel.as_matrix() )
        F1 = metrics.f1_score(np.array(testLabel), machine.predict(test.as_matrix()).round())

        # Grab the F1 scores for graphing
        x.append(len(features))
        y.append(F1)
        
        # Update condition variables
        old = new
        new = F1
    
    # Graph the F1 scores as a function
    plt.plot(x, y, '-')
    plt.title('Incrementally Adding Best Independent Features')
    plt.ylabel('F1 Score')
    plt.xlabel('Number of Features')
    plt.show()
    
    # As a bar chart
    plt.bar(x, y)
    plt.title('Incrementally Adding Best Independent Features')
    plt.ylabel('F1 Score')
    plt.xlabel('Number of Features')
    plt.show()
    
    features.pop()
    return features

def feature_selection_best_dep(train_d, test_d, features, threshold, x, y):
    pq = []

    # Rank the features in order of F1 score when operating with current model
    for feature in train_d.columns:
        # Don't double feature
        if feature in features:
            continue
        
        # Add a feature to the model
        features.append(feature)
        train = train_d[features]
        test = test_d[features]
        
        # Uncomment a machine
        #machine = tree.DecisionTreeRegressor( criterion="mse" )
        machine = naive_bayes.GaussianNB()
        #machine = KNeighborsClassifier(p=len(training.columns.values)*2)
        #machine = tree.DecisionTreeClassifier( criterion="entropy" )
        
        # Determine its F1 score
        F1 = 0
        
        if len(features) == 1:
            machine.fit(train.as_matrix().reshape(-1,1), trainLabel.as_matrix() )
            F1 = metrics.f1_score(np.array(testLabel), machine.predict(test.as_matrix().reshape(-1,1)).round())
        else:
            machine.fit(train.as_matrix(), trainLabel.as_matrix() )
            F1 = metrics.f1_score(np.array(testLabel), machine.predict(test.as_matrix()).round())

        
        # Throw that feature away
        features.pop()
        
        # If the model is an improvement, push the new feature onto the pq
        #if -F1 < threshold:    
        heapq.heappush(pq, (-F1, feature))
    
    # If pq is empty, adding additional features don't improve the model
    if not pq:
        return features

    # Take the feature that gives the best improvement and build off of that
    chosen = heapq.heappop(pq)
    features.append(chosen[1])
    threshold = chosen[0]
    
    print(chosen, threshold)
    
    # Store number of features and F1 score
    x.append(len(features))
    y.append(-threshold)
    
    # Recursive call to find another feature
    return feature_selection_best_dep(train_d, test_d, features, threshold, x, y)
    
    


def createDTree(training, testing):
    feature_selection_best_ind(training, testing)
    feature_selection_best_dep(training, testing, [], 0)


print(feature_selection_best_ind(training, testing))
print("")

x = []
y = []
print(feature_selection_best_dep(training, testing, [], 0, x, y))
print("")

# Graph the F1 scores as a function
plt.plot(x, y, '-')
plt.title('Incrementally Adding Best Dependent Features')
plt.ylabel('F1 Score')
plt.xlabel('Number of Features')
plt.show()

# As a bar chart
plt.bar(x, y)
plt.title('Incrementally Adding Best Dependent Features')
plt.ylabel('F1 Score')
plt.xlabel('Number of Features')
plt.show()

'''
# This is templates ignore it
print("# This is templates ignore it")
print(type(machine))
print("Prediction:", machine.predict( testing.as_matrix() ).round())
print("Real Value:", np.array(testLabel).round())

print("Confusion Matrix:\n", metrics.confusion_matrix( np.array(testLabel), machine.predict( testing.as_matrix() ).round()))
print("Accuracy:", metrics.accuracy_score( np.array(testLabel), machine.predict( testing.as_matrix() ).round()))
print("F1 Score:", metrics.f1_score( np.array(testLabel), machine.predict( testing.as_matrix() ).round()))
print("Precision:", metrics.precision_score( np.array(testLabel), machine.predict( testing.as_matrix() ).round()))
print("Recall:", metrics.recall_score( np.array(testLabel), machine.predict( testing.as_matrix() ).round()))


graph = Source( tree.export_graphviz(machine, out_file=None, feature_names=training.columns))
png_bytes = graph.pipe(format='png')
with open('dtree_pipe_proposed_model.png','wb') as f:
    f.write(png_bytes)

from IPython.display import Image
Image(png_bytes) 
'''