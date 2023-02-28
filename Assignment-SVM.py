# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 08:11:20 2023

@author: shrut
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.datasets import make_circles
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


# Create a dataset

X,y = make_circles(n_samples=1000, noise=.5, random_state=1234)
X.shape
np.unique(y, return_counts=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)


# ======================================
# Logistic Regression 
# ======================================

# Divide the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.3, random_state=1234, stratify=y)
# Build a logistic regression model and fit it to the data
clr = LogisticRegression(random_state=0).fit(X_train, y_train)
# Predict the y value for X_test
y_pred = clr.predict(X_test)

print(f'accuracy from training data: {clr.score(X_train,y_train):.2f}')
print(f'accuracy from testing data: {clr.score(X_test,y_test):.2f}')

# =====================================
# Gradient Boosting for Classification
# =====================================


gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

y_pred2 = gbc.predict(X_test)


print(f'accuracy from training data: {gbc.score(X_train,y_train):.2f}')
print(f'accuracy from testing data: {gbc.score(X_test,y_test):.2f}')

# =================================
# Support Vector Machines 
# =================================


svc = SVC(kernel='linear',random_state=1234).fit(X_train, y_train)
print(f'accuracy from training data: {svc.score(X_train,y_train):.2f}')
print(f'accuracy from testing data: {svc.score(X_test,y_test):.2f}')


# Try different kernels: 'poly','sigmoid', and 'rbf'.

for k in ('poly','sigmoid','rbf'):

    svc = SVC(kernel=k, random_state=1234).fit(X_train,y_train)
    print(f'--- kernel used: {k}---')
    print(f'accuracy from training data: {svc.score(X_train,y_train):.2f}')
    print(f'accuracy from testing data: {svc.score(X_test,y_test):.2f}\n')


# C: Regularization parameter, default=1. Try for different C values 

for c in (0.5,1,3,10,100):
    svc = SVC(kernel='rbf',C=c , random_state=1234).fit(X_train,y_train)
    print('--- C for rbf ---')
    print (f'C = {c}')
    print(f'accuracy from training data: {svc.score(X_train,y_train):.2f}')
    print(f'accuracy from testing data: {svc.score(X_test,y_test):.2f}\n')

    
# Gamma: Try for different gamma values 


for g in (.05,.1,.3,1,3):
    svc = SVC(kernel='rbf', gamma=g, random_state=1234).fit(X_train,y_train)
    print('--- gamma for rbf ---')
    print (f'gamma = {g}')
    print(f'accuracy from training data: {svc.score(X_train,y_train):.2f}')
    print(f'accuracy from testing data: {svc.score(X_test,y_test):.2f}\n')


