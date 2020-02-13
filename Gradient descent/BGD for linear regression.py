# import relevant packages
import numpy as np
import sklearn

# in many cases the the normal equation will be sufficient for the learning required, however, then the number of
# training examples gets very large (in terms of features), NE becomes too computationally expensive


# Batch gradient descent Implementing BGD requires taking the cost function's derivative with respect to the
# parameter in question (i.e. partial derivative)

# Data
np.random.seed(150)

X = 2 * np.random.rand(100, 1)
y = 20 + 7 * X + np.random.randn(100, 1)  # y is 20 + 7 * X  + noise
X_beta = np.c_[np.ones((100, 1)), X]  # add the intercept of 1 to each observation
theta_best = np.linalg.inv(X_beta.T.dot(X_beta)).dot(X_beta.T).dot(
    y)  # we compute the inverse of the matrix using the lin alg inv function, the matrix is the dot product of

# Predictions
X_new = np.array([[0], [2]])  #
X_new_b = np.c_[np.ones((2, 1)), X_new]  # append to X_new

# SGD
eta = 0.1  # learning rate
n_iterations = 1000  # iterations for the bdg
m = 100

theta = np.random.randn(2, 1)  # random initialization

for iteration in range(n_iterations):
    gradients = 2 / m * X_beta.T.dot(X_beta.dot(theta) - y)
    theta = theta - eta * gradients

preds_BGD = X_new_b.dot(theta)

print('theta bgd: ', theta)
print('preds: ', preds_BGD)
