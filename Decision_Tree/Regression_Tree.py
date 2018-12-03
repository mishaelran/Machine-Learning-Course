"""
Created on Sun Dec  2 13:48:32 2018

@author: rmishael
"""
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values


#create your reggressor here:
from sklearn.tree import DecisionTreeRegressor as DTR
regressor = DTR(random_state = 0)
regressor.fit(X,y)


#visualising the  regression results
plt.scatter(X, y, color = 'pink')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Salary vs Position (Regression model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualising the  regression results with higer resolution
X_grid = np.arange(min(X), max(X), 0.01) 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'pink')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Salary vs Position (Regression model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

y_pred = regressor.predict(6.5)

