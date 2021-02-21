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

class DTClassifier():
    def decisionTreeClassifier(self,dt_clf,x_train,x_test, y_train):
        #print(y_train.values.ravel())
        #dt_clf = DecisionTreeClassifier(random_state=0)
        dt_clf.fit(x_train, y_train.values.ravel())
        y_predict_train = dt_clf.predict(x_train)
        y_predict_test = dt_clf.predict(x_test)
        
        train_accuracy = accuracy_score(y_train.values,y_predict_train)
        test_accuracy = accuracy_score(y_test.values,y_predict_test)
        # -------------------------------
        return dt_clf,y_predict_train, y_predict_test, train_accuracy, test_accuracy


#==============================================================================
#Load data
#==============================================================================
datatest = Data()
#path = 'AFP300_nonAFP300_train_AACandDipeptide_twoSeg.csv'
path = 'Class_BanknoteAuth.csv'
#path = 'pima-indians-diabetes.csv'


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

DT1 = DTClassifier()
dt_clf = DecisionTreeClassifier(random_state=0)
dt_clf,y_predict_train, y_predict_test, train_accuracy, test_accuracy = DT1.decisionTreeClassifier(dt_clf,x_train,x_test, y_train)

print('Training accuracy is: ',train_accuracy)
print('Test accuracy is: ',test_accuracy)

print ('Depth is ',dt_clf.tree_.max_depth)
print ('Node number is ',dt_clf.tree_.node_count)
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
estimator = DecisionTreeClassifier(random_state=0,criterion='gini'),
X = x_train,
y = y_train, train_sizes = train_sizes, cv = 5,
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
plt.title('Learning curves for a decision tree (Default setting)', y = 1.03)
plt.legend()
#plt.ylim(0,40)
plt.grid(True)
plt.savefig('DT_sample_B_Learning_curves_for_a_decision_tree_Default_setting.png',dpi=600)

#==============================================================================
#Post pruning with cost complexity parameter (ccp)
#https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
#==============================================================================
path = dt_clf.cost_complexity_pruning_path(x_train, y_train.values.ravel())
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.grid(True)
plt.savefig('DT_sample_B_Post_pruning_Total_Impurity_vs_effective_alpha_for_training_set.png',dpi=600)

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(x_train, y_train.values.ravel())
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))

# print (ccp_alphas)
# print (impurities)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]

fig, ax = plt.subplots(2, 1)
ax[0].grid(True)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")

ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
ax[1].grid(True)
fig.tight_layout()

plt.savefig('DT_sample_B_Post_pruning_nodes_and_depth_vs_alpha.png',dpi=600)

#==============================================================================
#Gid search
#==============================================================================
parameters = {'criterion':['gini', 'entropy'], 'max_depth':[2, 4, 6, 8], 'ccp_alpha':[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]}
dt_clf = DecisionTreeClassifier(random_state=0)
gscv_dt = GridSearchCV(dt_clf, parameters, scoring='accuracy', cv=5)
gscv_dt_fit = gscv_dt.fit(x_train, y_train.values.ravel())
best_params = gscv_dt.best_params_
best_score = gscv_dt.best_score_

print ('Best parameters are: ',best_params)
print ('Best score is: ',best_score)

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

max_depth_range = np.linspace(2, 8, 7)
ccp_alpha_range = np.linspace(0, 0.07, 8)
param_grid = dict(max_depth=max_depth_range, ccp_alpha=ccp_alpha_range)
#cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
cv = 5
grid = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid=param_grid, cv=cv)
grid.fit(x_train, y_train.values.ravel())

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

scores = grid.cv_results_['mean_test_score'].reshape(len(ccp_alpha_range),
                                                     len(max_depth_range))

print ('Max score: ',np.max(scores))
print ('Min score: ',np.min(scores))

#plt.style.use('seaborn')
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.85, midpoint=0.92))
plt.xlabel('max_depth')
plt.ylabel('ccp_alpha')
plt.colorbar()
plt.xticks(np.arange(len(max_depth_range)), max_depth_range, rotation=45)
plt.yticks(np.arange(len(ccp_alpha_range)), ccp_alpha_range)
plt.title('Validation accuracy')
plt.grid(False)
plt.savefig('DT_sample_B_Grid_search.png',dpi=600)
plt.show()
#plt.style.use('default')

#==============================================================================
# Validation Curve 1
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================
dt_clf = DecisionTreeClassifier(random_state=0)
param_range = [2, 3, 4, 5, 6, 7, 8]
train_scores, test_scores = validation_curve(
    dt_clf, X = x_train, y = y_train, param_name="max_depth", param_range=param_range,
    scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(2)
plt.title("Validation Curve with Decision Tree (Accuracy VS Max Depth)")
plt.xlabel("Max depth")
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
plt.savefig('DT_sample_B_Validation_Curve_Max_depth.png',dpi=600)
plt.show()


#==============================================================================
# Validation Curve 2
#https://scikit-learn.org/stable/modules/learning_curve.html
#==============================================================================
dt_clf = DecisionTreeClassifier(random_state=0)
param_range = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
train_scores, test_scores = validation_curve(
    dt_clf, X = x_train, y = y_train, param_name="ccp_alpha", param_range=param_range,
    scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(3)
plt.title("Validation Curve with Decision Tree (Accuracy VS ccp alpha)")
plt.xlabel("ccp alpha")
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
plt.savefig('DT_sample_B_Validation_Curve_ccp_alpha.png',dpi=600)
plt.show()

#==============================================================================
#learning curve
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#==============================================================================

train_sizes = [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for i in range(1, len(train_sizes)):
    train_sizes[i] = math.floor(train_sizes[i] *y_train.size*4/5)
    
#print (train_sizes)

train_sizes, train_scores, validation_scores = learning_curve(
estimator = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=6, ccp_alpha=0),
X = x_train,
y = y_train, train_sizes = train_sizes, cv = 5,
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

fig = plt.figure(4)
#plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error', color="darkorange", lw=2)
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error', color="navy", lw=2)
plt.ylabel('Accuracy')
plt.xlabel('Training set size')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a decision tree (after hyper-parameter tuning)', y = 1.03)
plt.legend()
plt.grid(True)
plt.savefig('DT_sample_B_Learning_curves_for_after_hyper_parameter_tuning.png',dpi=600)

#==============================================================================
#Final prediction
#==============================================================================

print('\n', '-' * 50)
print('After hyperparameter tunning, criterion=gini, max_depth=6, ccp_alpha=0')

start_1 = time.time()
dt_clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=6, ccp_alpha=0)
dt_clf.fit(x_train, y_train.values.ravel())
end_1 = time.time()
print('Train time is: ',end_1 - start_1)

start_2 = time.time()
y_predict_train = dt_clf.predict(x_train)
end_2 = time.time()
print('Predict time for training set is: ',end_2 - start_2)

start_3 = time.time()
y_predict_test = dt_clf.predict(x_test)
end_3 = time.time()
print('Predict time for test set is: ',end_3 - start_3)

train_accuracy = accuracy_score(y_train.values,y_predict_train)
test_accuracy = accuracy_score(y_test.values,y_predict_test)

report = classification_report(y_test.values,y_predict_test)

print('Training accuracy is: ',train_accuracy)
print('Test accuracy is: ',test_accuracy)

print ('Depth is ',dt_clf.tree_.max_depth)
print ('Node number is ',dt_clf.tree_.node_count)
print ('Classification report:')
print (report)
print('\n', '-' * 50)


