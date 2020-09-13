###
##  Importing all the nencessary modules.

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

###
##  Importing dataset from sklearn
from sklearn.datasets import load_breast_cancer

###
##  Importing train_test_split
from sklearn.model_selection import train_test_split

###
## Importing support vector classifier
from sklearn.svm import SVC

###
## Importing classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

###
##  Loading the dataset

cancer_data = load_breast_cancer()

print(cancer_data.keys())

###
##  For dataset description
print(cancer_data['DESCR'])


###
##  Creating DataFrame object

df = pd.DataFrame(cancer_data['data'],columns = cancer_data['feature_names'])

print(df.head(5)) ## Print first five rows of dataset.


X = df
Y = cancer_data['target']

###
##  Splitting the data into train and test

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 44)

###
##  Creating SVM classifier object
svm_model = SVC(C = 1.0, kernel = 'rbf')

###
## Fitting the model with training data
svm_model.fit(X_train,Y_train)

###
##  Prediction
prediction = svm_model.predict(X_test)

###
## Classification report and confusion matrix

print(confusion_matrix(Y_test, prediction))
print(classification_report(Y_test, prediction))
