# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:56:22 2018

@author: Jojo
"""

from sklearn import svm, metrics, grid_search, tree
from sklearn.neighbors import KNeighborsClassifier
from pandas import DataFrame
import numpy as np

training = DataFrame.from_csv("../data/deltas_training.csv")
testing = DataFrame.from_csv("../data/deltas_testing.csv")
training = training.dropna(axis=1, how='all')
testing = testing.dropna(axis=1, how='all')
training = training.dropna(axis=0, how='any')
testing = testing.dropna(axis=0, how='any')
trainLabel = training['Label']
testLabel = testing['Label']
trainHomes = training['home_team']
trainAways = training['away_team']
testHomes = testing['home_team']
testAways = testing['away_team']

training = training.drop('away_team', 1).drop('home_team', 1).drop('neutral_site?', 1)
#training = training.drop("", 1)
testing = testing.drop('away_team', 1).drop('home_team', 1).drop('neutral_site?', 1)
#testing = testing.drop("year", 1)


def svc_param_selection(X, y, nfolds):
    Cs = [10, 1, 0.1, 0.2]
    gammas = [0.001, 0.01, 0.1, 1, 2, 4]
    param_grid = {'C': Cs, 'gamma' : gammas}
    g_search = grid_search.GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds, scoring='neg_mean_squared_error')
    g_search.fit(X, y)
    #print(g_search)
    g_search.best_params_
    #print(g_search.best_params_)
    return g_search.best_params_


#p = svc_param_selection( training.as_matrix(), trainLabel.as_matrix(), 3 )

#machine = svm.SVC( C=p['C'], gamma=p['gamma'])
#machine = KNeighborsClassifier(p=40)
machine = tree.DecisionTreeClassifier()
#print(machine)
machine.fit( training.as_matrix() , trainLabel.as_matrix() )
print(type(machine))
print("Prediction:", machine.predict( testing.as_matrix() ))
print("Real Value:", np.array(testLabel))

print("Confusion Matrix:\n", metrics.confusion_matrix( np.array(testLabel), machine.predict( testing.as_matrix() )))
print("Accuracy:", metrics.accuracy_score( np.array(testLabel), machine.predict( testing.as_matrix() )))
print("F1 Score:", metrics.f1_score( np.array(testLabel), machine.predict( testing.as_matrix() )))
print("Precision:", metrics.precision_score( np.array(testLabel), machine.predict( testing.as_matrix() )))
print("Recall:", metrics.recall_score( np.array(testLabel), machine.predict( testing.as_matrix() )))