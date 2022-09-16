import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("C:/Users/ahmet/OneDrive/Masaüstü/GitHub/Position_Salaries.csv")

x = data.iloc[:,1:2]
y = data.iloc[:,2:]

X = x.values
Y = y.values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()
lin_reg1.fit(X, Y)

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, Y)

#scaling
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
sc2 = StandardScaler()

x_scaled = sc1.fit_transform(X)
y_scaled = sc2.fit_transform(Y)

#support vector regression
from sklearn.svm import SVR
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_scaled, y_scaled.ravel())

#decision tree
from sklearn.tree import DecisionTreeRegressor
t_reg = DecisionTreeRegressor(random_state = 0)
t_reg.fit(X, Y)

#visualisation
plt.scatter(X, Y, color="r")
plt.plot(X, lin_reg1.predict(X), color="b")
plt.title("Linear Regression")
plt.grid(True)
plt.show()

plt.scatter(X, Y, color="r")
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color="b")
plt.title("Polynomial Regression")
plt.grid(True)
plt.show()

plt.scatter(x_scaled, y_scaled, color="r")
plt.plot(x_scaled, svr_reg.predict(x_scaled), color="b")
plt.title("Support Vector Regression")
plt.grid(True)
plt.show()

plt.scatter(X, Y, color="r")
plt.plot(X, t_reg.predict(X), color="b")
plt.title("Decision Tree")
plt.grid(True)