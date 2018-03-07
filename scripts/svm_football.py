# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:56:22 2018

@author: Jojo
"""

from sklearn import svm, tree, metrics
from pandas import DataFrame
import numpy as np

training = DataFrame.from_csv("./train2.txt", sep='\t')
testing = DataFrame.from_csv("./test2.txt", sep='\t')

trainLabel = training['Label']
testLabel = testing['Label']

#print(type(testLabel.as_matrix()))

training = training.drop("Date", 1) #remove as column type
training = training.drop("Opponent", 1) #remove as column type
training = training.drop("Label", 1) #remove as column type

testing = testing.drop("Date", 1) #remove as column type
testing = testing.drop("Opponent", 1) #remove as column type
testing = testing.drop("Label", 1) #remove as column type
#features = training.columns.values[0:len(training.columns.values)-1]

#print(testing, training)

#print( training.as_matrix() )

machine = svm.SVC()
machine.fit( training.as_matrix() , trainLabel.as_matrix() )
print("Prediction:", machine.predict( testing.as_matrix() ))
print("Real Value:", np.array(testLabel))

print("Confusion Matrix:\n", metrics.confusion_matrix( np.array(testLabel), machine.predict( testing.as_matrix() )))
print("Accuracy:", metrics.accuracy_score( np.array(testLabel), machine.predict( testing.as_matrix() )))
print("F1 Score:", metrics.f1_score( np.array(testLabel), machine.predict( testing.as_matrix() )))
print("Precision:", metrics.precision_score( np.array(testLabel), machine.predict( testing.as_matrix() )))
print("Recall:", metrics.recall_score( np.array(testLabel), machine.predict( testing.as_matrix() )))