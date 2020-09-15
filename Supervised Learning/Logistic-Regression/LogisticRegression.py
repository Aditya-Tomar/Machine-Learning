###
##  Import all the required modules

import numpy as np
import pandas as pd
import sklearn

###
##  Importing train_test_split, LogisticRegression model and metrices

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


###
## You can ignore this function
## This function is used to fill the null value of age column

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

    

###
##  Importing dataset

data = pd.read_csv(r"C:\Users\Asus\Downloads\original (1)\Refactored_Py_DS_ML_Bootcamp-master\13-Logistic-Regression\titanic_train.csv")


#print(data.head(5))    # Uncomment it. To check the dataset structure.

###
##  Removing the column from dataset having data in string type.

data.drop(['Sex','Embarked','Name','Ticket','PassengerId','Cabin'],axis=1,inplace=True)
data['Age'] = data[['Age','Pclass']].apply(impute_age,axis=1)


X = data.drop('Survived',axis = 1)
Y = data['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state= 42)


###
##  Creating instance of logistic regression model

log_model = LogisticRegression(C=1.0, max_iter =1000, multi_class='ovr',penalty='l2',verbose=0)

###
##  Fitting the model with X_train and Y_train

log_model.fit(X_train,Y_train)

predictions = log_model.predict(X_test)

#print(predictions)

print('Classification_report\n\n', classification_report(predictions, Y_test) )
print('Confusion_matrix\n\n', confusion_matrix( predictions, Y_test) )
