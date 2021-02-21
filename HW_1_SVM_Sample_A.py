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
SVM_clf = SVC(random_state=0, C=10, gamma=0.0001)
#DT_clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=5, ccp_alpha=0.02)
SVM_clf.fit(x_train, y_train.values.ravel())
y_predict_train = SVM_clf.predict(x_train)
y_predict_test = SVM_clf.predict(x_test)
        
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
SVM_clf = SVC(random_state=0)
train_sizes, train_scores, validation_scores = learning_curve(
estimator = SVM_clf,
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
plt.title('Learning curves for SVM (Default setting)', y = 1.03)
plt.legend()
#plt.ylim(0,40)
plt.grid(True)
plt.savefig('SVM_sample_A_Learning_curves_default_setting.png',dpi=600)


#==============================================================================
#Gid search
#==============================================================================
parameters = {'C':np.logspace(-2, 10, 13)}
SVM_clf = SVC(random_state=0, kernel='linear')
gscv_dt = GridSearchCV(SVM_clf, parameters, scoring='accuracy', cv=5)
gscv_dt_fit = gscv_dt.fit(x_train, y_train.values.ravel())
best_params = gscv_dt.best_params_
best_score = gscv_dt.best_score_

print ('Best parameters are: ',best_params)
print ('Best score is: ',best_score)
# Best parameters are:  {'C': 0.01}
# Best score is:  0.8104166666666668

#==============================================================================
parameters = {'C':np.logspace(-2, 10, 13), 'gamma':np.logspace(-9, 3, 13)}
SVM_clf = SVC(random_state=0, kernel='sigmoid')
gscv_dt = GridSearchCV(SVM_clf, parameters, scoring='accuracy', cv=5)
gscv_dt_fit = gscv_dt.fit(x_train, y_train.values.ravel())
best_params = gscv_dt.best_params_
best_score = gscv_dt.best_score_

print ('Best parameters are: ',best_params)
print ('Best score is: ',best_score)
# Best parameters are:  {'C': 10.0, 'gamma': 0.0001}
# Best score is:  0.8395833333333333

#==============================================================================
parameters = {'C':np.logspace(-2, 10, 13), 'gamma':np.logspace(-9, 3, 13)}
SVM_clf = SVC(random_state=0, kernel='rbf')
gscv_dt = GridSearchCV(SVM_clf, parameters, scoring='accuracy', cv=5)
gscv_dt_fit = gscv_dt.fit(x_train, y_train.values.ravel())
best_params = gscv_dt.best_params_
best_score = gscv_dt.best_score_

print ('Best parameters are: ',best_params)
print ('Best score is: ',best_score)
# Best parameters are:  {'C': 10.0, 'gamma': 0.0001}
# Best score is:  0.85

#==============================================================================
parameters = {'C':np.logspace(-2, 10, 13), 'gamma':np.logspace(-9, 3, 13), 'degree':[2,3,4,5]}
SVM_clf = SVC(random_state=0, kernel='poly')
gscv_dt = GridSearchCV(SVM_clf, parameters, scoring='accuracy', cv=5)
gscv_dt_fit = gscv_dt.fit(x_train, y_train.values.ravel())
best_params = gscv_dt.best_params_
best_score = gscv_dt.best_score_

print ('Best parameters are: ',best_params)
print ('Best score is: ',best_score)
# Best parameters are:  {'C': 0.01, 'degree': 2, 'gamma': 0.01}
# Best score is:  0.6416666666666666

#==============================================================================
#Grid search for Hyper parameters - Maxdepth and ccp_alpha
#https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
#==============================================================================
# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
#cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
cv = 5
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(x_train, y_train.values.ravel())

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                      len(gamma_range))

print ('Max score: ',np.max(scores))
print ('Min score: ',np.min(scores))

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
            norm=MidpointNormalize(vmin=0.4, midpoint=0.8))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.grid(False)
plt.savefig('SVM_sample_A_Grid_search.png',dpi=600)
plt.show()


#==============================================================================
# Validation Curve 1
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================
param_range = np.logspace(-9, 3, 13)
train_scores, test_scores = validation_curve(
    SVC(C=10), X = x_train, y = y_train.values.ravel(), param_name="gamma", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.figure(2)
plt.title("Validation Curve with SVM (Accuracy VS gamma)")
plt.xlabel("gamma")
plt.ylabel("Score")
#plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
              color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                  train_scores_mean + train_scores_std, alpha=0.2,
                  color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
              color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                  test_scores_mean + test_scores_std, alpha=0.2,
                  color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('SVM_sample_A_Validation_Curve_gamma.png',dpi=600)
plt.show()


#==============================================================================
# Validation Curve 2
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================
param_range = np.logspace(-2, 10, 13)
train_scores, test_scores = validation_curve(
    SVC(gamma=0.0001), X = x_train, y = y_train.values.ravel(), param_name="C", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.figure(3)
plt.title("Validation Curve with SVM (Accuracy VS Regularization parameter C)")
plt.xlabel("Regularization parameter C")
plt.ylabel("Score")
#plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
              color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                  train_scores_mean + train_scores_std, alpha=0.2,
                  color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
              color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                  test_scores_mean + test_scores_std, alpha=0.2,
                  color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('SVM_sample_A_Validation_Curve_C.png',dpi=600)
plt.show()



#==============================================================================
#learning curve 4
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#==============================================================================
train_sizes = [10, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for i in range(1, len(train_sizes)):
    train_sizes[i] = math.floor(train_sizes[i] *y_train.size*4/5)
    
#print (train_sizes)
SVM_clf = SVC(random_state=0, C=10, gamma=0.0001)
train_sizes, train_scores, validation_scores = learning_curve(
estimator = SVM_clf,
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
plt.title('Learning curves for SVM (After hyper-parameter tuning)', y = 1.03)
plt.legend()
#plt.ylim(0,40)
plt.grid(True)
plt.savefig('SVM_sample_A_Learning_curves_after_hyper_parameter_tuning.png',dpi=600)

#==============================================================================
#Final prediction
#==============================================================================

print('\n', '-' * 50)
print('After hyperparameter tunning, C=10, gamma=0.0001')

start_1 = time.time()
SVM_clf = SVC(random_state=0, C=10, gamma=0.0001)
SVM_clf.fit(x_train, y_train.values.ravel())
end_1 = time.time()
print('Train time is: ',end_1 - start_1)

start_2 = time.time()
y_predict_train = SVM_clf.predict(x_train)
end_2 = time.time()
print('Predict time for training set is: ',end_2 - start_2)

start_3 = time.time()
y_predict_test = SVM_clf.predict(x_test)
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







