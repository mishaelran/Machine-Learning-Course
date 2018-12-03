
"""
Created on Sun Dec  2 14:02:00 2018

@author: rmishael
"""
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_Y.fit_transform(y)

#create your reggressor here:
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#fitting back the scale for real slaries:


#predicting a new result
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#visualising the  regression results
plt.scatter(X, y, color = 'pink')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Salary vs Position (Regression model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualising the  regression results with higer resolution
X_grid = np.arange(min(X), max(X), 0.1) 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'pink')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Salary vs Position (Regression model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



