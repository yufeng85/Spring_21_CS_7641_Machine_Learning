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

MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic',max_iter = 500,alpha = 1e-4,hidden_layer_sizes = (100),random_state = 0)
MLP_clf.fit(x_train, y_train.values.ravel())
y_predict_train = MLP_clf.predict(x_train)
y_predict_test = MLP_clf.predict(x_test)
        
train_accuracy = accuracy_score(y_train.values,y_predict_train)
test_accuracy = accuracy_score(y_test.values,y_predict_test)

print('Training accuracy is: ',train_accuracy)
print('Test accuracy is: ',test_accuracy)
print('\n', '-' * 50)

#==============================================================================
#learning curve
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#==============================================================================
train_sizes = [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for i in range(1, len(train_sizes)):
    train_sizes[i] = math.floor(train_sizes[i] *y_train.size*4/5)
    
#print (train_sizes)

train_sizes, train_scores, validation_scores = learning_curve(
estimator = MLPClassifier(solver='sgd',activation = 'logistic',max_iter = 1000,alpha = 1e-4,hidden_layer_sizes = (100),random_state = 0),
X = x_train,
y = y_train.values.ravel(), train_sizes = train_sizes, cv = 5,
scoring = 'accuracy',
shuffle = True,
random_state=0)


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
plt.title('Learning curves for ANN (Default setting)', y = 1.03)
plt.legend()
#plt.ylim(0,40)
plt.grid(True)
plt.savefig('ANN_sample_A_Learning_curves_for_ANN_Default_setting.png',dpi=600)


#==============================================================================
#Gid search
#==============================================================================
# parameters = {'criterion':['gini', 'entropy']}
# dt_clf = DecisionTreeClassifier(random_state=0)
# gscv_dt = GridSearchCV(dt_clf, parameters, scoring='accuracy', cv=5)
# gscv_dt_fit = gscv_dt.fit(x_train, y_train.values.ravel())
# best_params = gscv_dt.best_params_
# best_score = gscv_dt.best_score_

# print ('Best parameters are: ',best_params)
# print ('Best score is: ',best_score)

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

#hidden_layer_sizes_range = np.linspace(0, 400, 9)
#hidden_layer_sizes_range[0] = 10
hidden_layer_sizes_range = [(10), (50), (100), (150), (200), (250), (300), (350), (400)]
alpha_range = [1e-2, 1e-3, 1e-4, 1e-5]
param_grid = dict(hidden_layer_sizes=hidden_layer_sizes_range, alpha=alpha_range)
#cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
cv = 5
grid = GridSearchCV(MLPClassifier(solver='sgd',activation = 'logistic',max_iter = 1000,random_state = 0), param_grid=param_grid, cv=cv)
grid.fit(x_train, y_train.values.ravel())

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

scores = grid.cv_results_['mean_test_score'].reshape(len(alpha_range),
                                                      len(hidden_layer_sizes_range))

print ('Max score: ',np.max(scores))
print ('Min score: ',np.min(scores))

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
            norm=MidpointNormalize(vmin=0.835, midpoint=0.847))
plt.xlabel('hidden_layer_sizes')
plt.ylabel('alpha')
plt.colorbar()
plt.xticks(np.arange(len(hidden_layer_sizes_range)), hidden_layer_sizes_range, rotation=45)
plt.yticks(np.arange(len(alpha_range)), alpha_range)
plt.title('Validation accuracy')
plt.grid(False)
plt.savefig('ANN_sample_A_Grid_search.png',dpi=600)
plt.show()

#==============================================================================
# Validation Curve 1
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================
MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', alpha = 1e-4,hidden_layer_sizes = (10),random_state = 0)
param_range = [10, 20, 30, 40, 50, 100, 200, 400, 600, 800, 1000, 1200, 1800]
train_scores, test_scores = validation_curve(
    MLP_clf, X = x_train, y = y_train.values.ravel(), param_name="max_iter", param_range=param_range,
    scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(2)
plt.title("Accuracy VS Max Iteration")
plt.xlabel("Max iteration")
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
plt.savefig('ANN_sample_A_Validation_Curve_Max_iteration.png',dpi=600)
plt.show()


#==============================================================================
# Validation Curve 2
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================

MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 1500, alpha = 1e-4, random_state = 0)
param_range = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 50, 75, 100]
train_scores, test_scores = validation_curve(
    MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='hidden_layer_sizes', param_range=param_range,
    scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(2)
plt.title("Accuracy VS Hidden Layer Sizes (logistic)")
plt.xlabel("Hidden layer sizes")
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
plt.savefig('ANN_sample_A_Validation_Curve_hidden_layer_sizes_logistic.png',dpi=600)
plt.show()

#==============================================================================
# Validation Curve 3
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================

MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 1500, hidden_layer_sizes = 20, random_state = 0)
param_range = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
train_scores, test_scores = validation_curve(
    MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='alpha', param_range=param_range,
    scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(3)
plt.title("Accuracy VS Alpha")
plt.xlabel("Alpha")
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
plt.xscale('log')
plt.savefig('ANN_sample_A_Validation_Curve_alpha.png',dpi=600)
plt.show()

#==============================================================================
# Validation Curve 4
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================

MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 1500, alpha = 1e-4, random_state = 0)
param_range = [(20,),(20,2), (20,4), (20,6), (20,8), (20,10), (20,15), (20,20), (20,50), (20,100), (20,150)]
train_scores, test_scores = validation_curve(
    MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='hidden_layer_sizes', param_range=param_range,
    scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(4)
plt.title("Validation Curve with ANN")
plt.xlabel("Second hidden layer sizes")
plt.ylabel("Score")
#plt.ylim(0.0, 1.1)
lw = 2
plt.plot([0,2,4,6,8,10,15,20,50,100,150], train_scores_mean, label="Training score",
              color="darkorange", lw=lw)
plt.fill_between([0,2,4,6,8,10,15,20,50,100,150], train_scores_mean - train_scores_std,
                  train_scores_mean + train_scores_std, alpha=0.2,
                  color="darkorange", lw=lw)
plt.plot([0,2,4,6,8,10,15,20,50,100,150], test_scores_mean, label="Cross-validation score",
              color="navy", lw=lw)
plt.fill_between([0,2,4,6,8,10,15,20,50,100,150], test_scores_mean - test_scores_std,
                  test_scores_mean + test_scores_std, alpha=0.2,
                  color="navy", lw=lw)
plt.legend(loc="best")
plt.grid(True)
#plt.xscale('log')
plt.savefig('ANN_sample_A_Validation_Curve_second_hidden_layer_sizes.png',dpi=600)
plt.show()

#==============================================================================
# Validation Curve 5
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================

MLP_clf = MLPClassifier(solver='sgd',activation = 'tanh', max_iter = 1500, alpha = 1e-4, random_state = 0)
param_range = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 50, 75, 100, 150, 200]
train_scores, test_scores = validation_curve(
    MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='hidden_layer_sizes', param_range=param_range,
    scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(5)
plt.title("Validation Curve with ANN (tanh)")
plt.xlabel("Hidden layer sizes")
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
plt.savefig('ANN_sample_A_Validation_Curve_hidden_layer_sizes_tanh.png',dpi=600)
plt.show()

#==============================================================================
# Validation Curve 6
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================

MLP_clf = MLPClassifier(solver='sgd',activation = 'relu', max_iter = 1500, alpha = 1e-4, random_state = 0)
param_range = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 50, 75, 100, 150, 200]
train_scores, test_scores = validation_curve(
    MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='hidden_layer_sizes', param_range=param_range,
    scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(6)
plt.title("Validation Curve with ANN (relu)")
plt.xlabel("Hidden layer sizes")
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
plt.savefig('ANN_sample_A_Validation_Curve_hidden_layer_sizes_relu.png',dpi=600)
plt.show()

#==============================================================================
# Validation Curve 7
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================

MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 3000, hidden_layer_sizes = 20, random_state = 0)
param_range = [1e-5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99999]
train_scores, test_scores = validation_curve(
    MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='momentum', param_range=param_range,
    scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(7)
plt.title("Accuracy VS Momentum (logistic)")
plt.xlabel("Momentum")
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
plt.savefig('ANN_sample_A_Validation_Curve_momentum.png',dpi=600)
plt.show()

#==============================================================================
# Momentum Curve 
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================
clfs = []
iters = []
momentums = [1e-5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99999]

for momentum in momentums:
    MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 3000, hidden_layer_sizes = 20, random_state = 0, momentum=momentum)
    MLP_clf.fit(x_train, y_train.values.ravel())
    clfs.append(MLP_clf)
    iters.append(MLP_clf.n_iter_)

plt.figure(8)
plt.title("Momentum VS Iteration")
plt.xlabel("Momentum")
plt.ylabel("Iterations needed for converge")
#plt.ylim(0.0, 1.1)
lw = 2
plt.plot(momentums, iters,
              color="navy", lw=lw)
#plt.legend(loc="best")
plt.grid(True)
plt.savefig('ANN_sample_A_momentum_VS_Iteration.png',dpi=600)
plt.show()


#==============================================================================
# Learning rate Curve 1
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================
clfs = []
iters = []
learning_rate_inits = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

for learning_rate_init in learning_rate_inits:
    MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 4000, hidden_layer_sizes = 20, random_state = 0, learning_rate_init=learning_rate_init)
    MLP_clf.fit(x_train, y_train.values.ravel())
    clfs.append(MLP_clf)
    iters.append(MLP_clf.n_iter_)

plt.figure(9)
plt.title("Validation Curve with ANN (logistic)")
plt.xlabel("Learning_rate_init")
plt.ylabel("Iterations needed for converge")
#plt.ylim(0.0, 1.1)
lw = 2
plt.plot(learning_rate_inits, iters,
              color="navy", lw=lw)
#plt.legend(loc="best")
plt.grid(True)
plt.xscale('log')
plt.savefig('ANN_sample_A_learning_rate_init_VS_Iteration.png',dpi=600)
plt.show()


#==============================================================================
# Validation Curve 7
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================

MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 3000, hidden_layer_sizes = 20, random_state = 0)
param_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
train_scores, test_scores = validation_curve(
    MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='learning_rate_init', param_range=param_range,
    scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(10)
plt.title("Validation Curve with ANN (logistic)")
plt.xlabel("Learning_rate_init")
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
plt.xscale('log')
plt.savefig('ANN_sample_A_Validation_Curve_learning_rate_init.png',dpi=600)
plt.show()


#==============================================================================
#learning curve 2
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#==============================================================================
train_sizes = [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for i in range(1, len(train_sizes)):
    train_sizes[i] = math.floor(train_sizes[i] *y_train.size*4/5)
    
#print (train_sizes)

train_sizes, train_scores, validation_scores = learning_curve(
estimator = MLPClassifier(solver='sgd',activation = 'logistic',max_iter = 800,alpha = 1e-4,hidden_layer_sizes = (20),random_state = 0),
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

fig = plt.figure(10)
#plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error', color="darkorange", lw=2)
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error', color="navy", lw=2)
plt.ylabel('Accuracy')
plt.xlabel('Training set size')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for ANN (after hyper-parameter tuning)', y = 1.03)
plt.legend()
#plt.ylim(0,40)
plt.grid(True)
plt.savefig('ANN_sample_A_Learning_curves_for_ANN_after_hyper_parameter_tunning.png',dpi=600)

#==============================================================================
#Final prediction
#==============================================================================

print('\n', '-' * 50)
print('After hyperparameter tunning, max_iter = 800, alpha = 1e-4,hidden_layer_sizes = (20)')

start_1 = time.time()
MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic',max_iter = 800,alpha = 1e-4,hidden_layer_sizes = (20),random_state = 0)
MLP_clf.fit(x_train, y_train.values.ravel())
end_1 = time.time()
print('Train time is: ',end_1 - start_1)

start_2 = time.time()
y_predict_train = MLP_clf.predict(x_train)
end_2 = time.time()
print('Predict time for training set is: ',end_2 - start_2)

start_3 = time.time()
y_predict_test = MLP_clf.predict(x_test)
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
