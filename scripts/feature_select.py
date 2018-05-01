# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:56:22 2018

@author: Jojo
"""

import matplotlib.pyplot as plt
from graphviz import Source
from sklearn import svm, metrics, grid_search, tree
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.datasets import load_iris
from pandas import DataFrame
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier


training = DataFrame.from_csv("../data/raw_data_normalized_t.csv")
testing = DataFrame.from_csv("../data/raw_data_normalized_test.csv")
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

training = training.drop('away_team', 1).drop('home_team', 1).drop("Label", 1)
#training = training.drop("", 1)
testing = testing.drop('away_team', 1).drop('home_team', 1).drop("Label", 1)
#testing = testing.drop("year", 1)

iris = load_iris()


def svc_param_selection(X, y, nfolds):
    Cs = [ 10, 7.5, 5, 2.5, 1, 0.1, 0.01]
    gammas = [ .125, .175, .25, 0.35, .5, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    g_search = grid_search.GridSearchCV(svm.SVR(kernel='poly', degree=3), param_grid, cv=nfolds, scoring='neg_mean_squared_error')
    g_search.fit(X, y)
    #print(g_search)
    g_search.best_params_
    #print(g_search.best_params_)
    return g_search.best_params_

dropped = list()

def feature_select(train_d, test_d, accuracy):
    print(" [ RUNNING ] NEW FUNCTION CALL feature_select")
    accuracy_t = accuracy
    flag = False
    for i in train_d.columns:
        train_t = train_d[i]
        test_t = test_d[i]
        #print(train_t, test_t)
        train_d.drop(i, axis=1, inplace=True)
        test_d.drop(i, axis=1, inplace=True)
        #print(train_d[i])
        #p = svc_param_selection( train_d.as_matrix(), trainLabel.as_matrix(), 10 )
        
        '''
        #machine = svm.SVR( C=p['C'], gamma=p['gamma'], kernel='poly', degree=3)
        machine = grid_search.GridSearchCV( svm.SVR( kernel='poly', degree=3 ), cv=5, param_grid = {"C": [ 2, 1, 0.1, 0.01, 0.001], "gamma": np.logspace(-2, 2, 5)})
        machine.fit( train_d.as_matrix() , trainLabel.as_matrix() )
        anew = metrics.accuracy_score( np.array(testLabel), machine.predict( test_d.as_matrix() ).round())
        print ("SVR poly3: ", anew)
        '''

        machine = grid_search.GridSearchCV( svm.SVR( kernel='rbf', degree=3 ), cv=5, param_grid = {"C": [ 2, 1, 0.1, 0.01, 0.001], "gamma": np.logspace(-2, 2, 5)})
        machine.fit( train_d.as_matrix() , trainLabel.as_matrix() )
        anew = metrics.f1_score( np.array(testLabel), machine.predict( test_d.as_matrix() ).round())
        print ("SVR rbf  : ", anew)
        
        '''
        machine = tree.DecisionTreeRegressor( criterion="mse" )
        machine.fit( train_d.as_matrix() , trainLabel.as_matrix() )
        anew = metrics.accuracy_score( np.array(testLabel), machine.predict( test_d.as_matrix() ).round())
        print ("ID3      : ", anew)
        '''
        
        print("[ TESTING ]", i, anew, ", RUN Number: ",len(dropped)-int(flag)+1)
        if anew > accuracy_t:
            accuracy_t = anew
            print("[ DROPPED ]", i, accuracy_t)
            best_t = i
            dropped.append(i)
            flag = True
        
        train_d[i] = train_t
        test_d[i] = test_t
    
    if (flag): 
        feature_select(train_d.drop(best_t), test_d.drop(best_t), accuracy_t)
    else:
        print (dropped)
    
    
    

#feature_select(training,testing,0)
'''
p = svc_param_selection( training.as_matrix(), trainLabel.as_matrix(), 10 )

fig, ax = plt.subplots(figsize=(35,35)) 
sns.heatmap(training.corr(), xticklabels=training.corr().columns.values, yticklabels=training.corr().columns.values, ax=ax)

#print(training.corr())
machine = svm.SVR( C=p['C'], gamma=p['gamma'])
machine = grid_search.GridSearchCV( svm.SVR( kernel='rbf', degree=3 ), cv=2, param_grid = {"C": [ 2, 1, 0.1, 0.01, 0.001], "gamma": np.logspace(-2, 2, 5)})


#machine = KNeighborsClassifier(p=len(training.columns.values)*2)
#machine = tree.DecisionTreeClassifier( criterion="entropy" )
print(machine)

machine.fit( training.as_matrix() , trainLabel.as_matrix().round() )
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

model = ExtraTreesClassifier()
model.fit(training.as_matrix(), trainLabel.as_matrix())
#print(sorted(model.feature_importances_))


model = ExtraTreesClassifier()
rfe = RFE(model, 14)
fit = rfe.fit(training.as_matrix(), trainLabel.as_matrix())
#print("Num Features: %d") % fit.n_features_
#print("Selected Features: %s") % fit.support_
#print("Feature Ranking: %s") % fit.ranking_
print(fit.support_)
print(fit.ranking_)


dummy = DataFrame()

for i in range(len(fit.ranking_)):
    if fit.ranking_[i] == 1:
        #print(list(training.columns)[i], fit.support_[i])
        dummy[list(training.columns)[i]] = training[list(training.columns)[i]]
        
dummy['Label'] = trainLabel

#print(dummy)
        
print(dummy.corr())

fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(dummy.corr(), annot=True, xticklabels=dummy.corr().columns.values, yticklabels=dummy.corr().columns.values, ax=ax)

print(training.corr())

    