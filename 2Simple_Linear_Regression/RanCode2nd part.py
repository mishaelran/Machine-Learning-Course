import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#Splitting the dataset into the training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Simple linear Regression to training set
from sklearn.linear_model import LinearRegression as LR
regressor = LR()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title('Salary vs Experiece (Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'pink')
plt.title('Salary VS Experience (Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()