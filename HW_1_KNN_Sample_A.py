# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 21:00:52 2021

@author: vince
"""

import numpy as np
import pandas as pd
import time
import gc
import random
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import validation_curve
from matplotlib.colors import Normalize
from sklearn.model_selection import StratifiedShuffleSplit
import timeit
from  sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

class Data():
    
    # points [1]
    def dataAllocation(self,path):
        # Separate out the x_data and y_data and return each
        # args: string path for .csv file
        # return: pandas dataframe, pandas dataframe
        data = pd.read_csv(path)
        xList = [i for i in range(data.shape[1] - 1)]
        x_data = data.iloc[:,xList]
        y_data = data.iloc[:,[-1]]
        # ------------------------------- 
        return x_data,y_data
    
    # points [1]
    def trainSets(self,x_data,y_data):
        # Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
        # Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 614.
        # args: pandas dataframe, pandas dataframe
        # return: pandas dataframe, pandas dataframe, pandas series, pandas series

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=614, shuffle=True)       
        # -------------------------------
        return x_train, x_test, y_train, y_test


#==============================================================================
#Load data
#==============================================================================
datatest = Data()
#path = 'Class_BanknoteAuth.csv'
#path = 'pima-indians-diabetes.csv'
path = 'AFP300_nonAFP300_train_AACandDipeptide_twoSeg.csv'

x_data,y_data = datatest.dataAllocation(path)
print("dataAllocation Function Executed")

x_train, x_test, y_train, y_test = datatest.trainSets(x_data,y_data)
print("trainSets Function Executed")


n = 0
for i in range(y_train.size):
    n = n + y_train.iloc[i,0]
print ('Positive rate for train data is: ',n/y_train.size)

n = 0
for i in range(y_test.size):
    n = n + y_test.iloc[i,0]
print ('Positive rate for test data is: ',n/y_test.size)

#Pre-process the data to standardize it
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#==============================================================================
#Default setting
#==============================================================================
plt.style.use('default')
print('\n', '-' * 50)
print('Default setting')
KNN_clf = KNeighborsClassifier()
#DT_clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=5, ccp_alpha=0.02)
KNN_clf.fit(x_train, y_train.values.ravel())
y_predict_train = KNN_clf.predict(x_train)
y_predict_test = KNN_clf.predict(x_test)
        
train_accuracy = accuracy_score(y_train.values,y_predict_train)
test_accuracy = accuracy_score(y_test.values,y_predict_test)

print('Training accuracy is: ',train_accuracy)
print('Test accuracy is: ',test_accuracy)
print('\n', '-' * 50)

#==============================================================================
#learning curve
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#==============================================================================
train_sizes = [10, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for i in range(1, len(train_sizes)):
    train_sizes[i] = math.floor(train_sizes[i] *y_train.size*4/5)
    
#print (train_sizes)
KNN_clf = KNeighborsClassifier()
train_sizes, train_scores, validation_scores = learning_curve(
estimator = KNN_clf,
X = x_train,
y = y_train.values.ravel(), train_sizes = train_sizes, cv = 5,
scoring = 'accuracy',
shuffle = True,
random_state=0)


#print('Training scores:\n\n', train_scores)
#print('\n', '-' * 70) # separator to make the output easy to read
#print('\nValidation scores:\n\n', validation_scores)

train_scores_mean = train_scores.mean(axis = 1)
validation_scores_mean = validation_scores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

fig = plt.figure(1)
#plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error', color="darkorange", lw=2)
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error', color="navy", lw=2)
plt.ylabel('Accuracy')
plt.xlabel('Training set size')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for KNN (Default setting)', y = 1.03)
plt.legend()
#plt.ylim(0,40)
plt.grid(True)
plt.savefig('KNN_sample_A_Learning_curves_default_setting.png',dpi=600)


#==============================================================================
#Gid search
#==============================================================================
parameters = {'weights':['uniform', 'distance'],'n_neighbors':[1,2,3,4,5,10,20,30,40,50,100,150,200,250,300,350]}
KNN_clf = KNeighborsClassifier()
gscv_dt = GridSearchCV(KNN_clf, parameters, scoring='accuracy', cv=5)
gscv_dt_fit = gscv_dt.fit(x_train, y_train.values.ravel())
best_params = gscv_dt.best_params_
best_score = gscv_dt.best_score_

print ('Best parameters are: ',best_params)
print ('Best score is: ',best_score)

#==============================================================================
parameters = {'weights':['uniform', 'distance'],'n_neighbors':[1,2,3,4,5,10,20,50,100,150,200],'p':[1,2,3,5,10,100,1000]}
KNN_clf = KNeighborsClassifier()
gscv_dt = GridSearchCV(KNN_clf, parameters, scoring='accuracy', cv=5)
gscv_dt_fit = gscv_dt.fit(x_train, y_train.values.ravel())
best_params = gscv_dt.best_params_
best_score = gscv_dt.best_score_

print ('Best parameters are: ',best_params)
print ('Best score is: ',best_score)



#==============================================================================
# Validation Curve 1
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================
param_range = [1,2,3,4,5,10,20,30,40,50,100,150,200,250]
train_scores, test_scores = validation_curve(
    KNeighborsClassifier(weights='uniform',p=1), X = x_train, y = y_train.values.ravel(), param_name="n_neighbors", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.figure(2)
plt.title("Validation Curve with KNN uniform (Accuracy VS n_neighbors)")
plt.xlabel("n_neighbors")
plt.ylabel("Score")
#plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
              color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                  train_scores_mean + train_scores_std, alpha=0.2,
                  color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
              color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                  test_scores_mean + test_scores_std, alpha=0.2,
                  color="navy", lw=lw)
plt.legend(loc="best")
plt.grid(True)
plt.savefig('KNN_sample_A_Validation_Curve_n_neighbors.png',dpi=600)
plt.show()


#==============================================================================
# Validation Curve 2
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================
param_range = [1,2,3,4,5,10,20,30,40,50,100,150,200,250]
train_scores, test_scores = validation_curve(
    KNeighborsClassifier(weights='distance',p=1), X = x_train, y = y_train.values.ravel(), param_name="n_neighbors", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.figure(2)
plt.title("Validation Curve with KNN distance (Accuracy VS n_neighbors)")
plt.xlabel("n_neighbors")
plt.ylabel("Score")
#plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
              color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                  train_scores_mean + train_scores_std, alpha=0.2,
                  color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
              color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                  test_scores_mean + test_scores_std, alpha=0.2,
                  color="navy", lw=lw)
plt.legend(loc="best")
plt.grid(True)
plt.savefig('KNN_sample_A_Validation_Curve_n_neighbors_distance.png',dpi=600)
plt.show()

#==============================================================================
# Validation Curve 3
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================
param_range = [1,2,3,4,5,6,7,8,9,10,20,30,40,50]
train_scores, test_scores = validation_curve(
    KNeighborsClassifier(weights='distance',n_neighbors=3), X = x_train, y = y_train.values.ravel(), param_name="p", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.figure(3)
plt.title("Validation Curve with KNN distance (Accuracy VS p)")
plt.xlabel("p")
plt.ylabel("Score")
#plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
              color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                  train_scores_mean + train_scores_std, alpha=0.2,
                  color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
              color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                  test_scores_mean + test_scores_std, alpha=0.2,
                  color="navy", lw=lw)
plt.legend(loc="best")
plt.grid(True)
plt.savefig('KNN_sample_A_Validation_Curve_p.png',dpi=600)
plt.show()



#==============================================================================
#learning curve 4
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#==============================================================================
train_sizes = [10, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for i in range(1, len(train_sizes)):
    train_sizes[i] = math.floor(train_sizes[i] *y_train.size*4/5)
    
#print (train_sizes)
KNN_clf = KNeighborsClassifier(weights='uniform',n_neighbors=3,p=1)
train_sizes, train_scores, validation_scores = learning_curve(
estimator = KNN_clf,
X = x_train,
y = y_train.values.ravel(), train_sizes = train_sizes, cv = 5,
scoring = 'accuracy',
shuffle = True,
random_state=0)


#print('Training scores:\n\n', train_scores)
#print('\n', '-' * 70) # separator to make the output easy to read
#print('\nValidation scores:\n\n', validation_scores)

train_scores_mean = train_scores.mean(axis = 1)
validation_scores_mean = validation_scores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

fig = plt.figure(12)
#plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error', color="darkorange", lw=2)
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error', color="navy", lw=2)
plt.ylabel('Accuracy')
plt.xlabel('Training set size')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for KNN (After hyper-parameter tuning)', y = 1.03)
plt.legend()
#plt.ylim(0,40)
plt.grid(True)
plt.savefig('KNN_sample_A_Learning_curves__after_hyper_parameter_tuning.png',dpi=600)

#==============================================================================
#Final prediction
#==============================================================================

print('\n', '-' * 50)
print('After hyperparameter tunning, weights=uniform, n_neighbors=3,p=1')

start_1 = time.time()
KNN_clf = KNeighborsClassifier(weights='uniform',n_neighbors=3,p=1)
KNN_clf.fit(x_train, y_train.values.ravel())
end_1 = time.time()
print('Train time is: ',end_1 - start_1)

start_2 = time.time()
y_predict_train = KNN_clf.predict(x_train)
end_2 = time.time()
print('Predict time for training set is: ',end_2 - start_2)

start_3 = time.time()
y_predict_test = KNN_clf.predict(x_test)
end_3 = time.time()
print('Predict time for test set is: ',end_3 - start_3)

train_accuracy = accuracy_score(y_train.values,y_predict_train)
test_accuracy = accuracy_score(y_test.values,y_predict_test)

report = classification_report(y_test.values,y_predict_test)

print('Training accuracy is: ',train_accuracy)
print('Test accuracy is: ',test_accuracy)

print ('Classification report:')
print (report)
print('\n', '-' * 50)







