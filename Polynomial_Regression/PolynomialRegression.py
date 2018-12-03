import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#linear regressor creation
from sklearn.linear_model import LinearRegression as LR
lin_regressor = LR()
lin_regressor.fit(X, y)

#Polynomial Regression creation
from sklearn.preprocessing import PolynomialFeatures as PF
poly_regressor = PF(degree = 4)
X_poly = poly_regressor.fit_transform(X)
lin_reg2 = LR()
lin_reg2.fit(X_poly, y)

#visualising the linear regression resutls
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_regressor.predict(X), color = 'green')
plt.title('Salary vs Position (Linear Regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()


#visualising the polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1) 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'pink')
plt.plot(X_grid, lin_reg2.predict(poly_regressor.fit_transform(X_grid)), color = 'blue')
plt.title('Salary vs Position (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting the new salary
lin_regressor.predict(6.5)
lin_reg2.predict(poly_regressor.fit_transform(6.5))