# import relevant packages
import numpy as np

# Standard linear regression models compute a linear function of a ste of predictor variables(x1...xn) on the outcome
# variable (y). This is computed by modelling a combined sum of each of the predictor variables with an additional
# bias term (intercept).
# This is generally summarised with this follow equation:
# y = mx + b or in the case of multivariate linear models: y= m1x1 + m2x2 +...... + mnxn + b.
# Where x is an input value, m is a parameter  and n is the number of features.
# A number of strategies potentially exist to train this algorithm and in this code i will implement them


# Normal equation
# theta hat = (X TRANSPOSE X)inx X t y

# Data
np.random.seed(150)

X = 2 * np.random.rand(100, 1)
y = 20 + 7 * X + np.random.randn(100, 1) # y is 20 + 7 * X  + noise
X_beta = np.c_[np.ones((100, 1)), X]  # add the intercept of 1 to each observation
theta_best = np.linalg.inv(X_beta.T.dot(X_beta)).dot(X_beta.T).dot(y) # we compute the inverse of the matrix using the lin alg inv function, the matrix is the dot product

# Predictions
X_new = np.array([[0], [2]]) #
X_new_b = np.c_[np.ones((2,1)), X_new] # append to X_new
predictions_y = X_new_b.dot(theta_best)

# what were the predictions
print('normal equation predictions: ',
      predictions_y)

# now lets compare to sklearn model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
predictions_s = lin_reg.predict(X_new)
print('sklearn predictions: ',
      predictions_s)




