# import relevant packages
import numpy as np
import sklearn

# Batch gradient descent is limited in the requirement to hold all of the data for each iteration and in this
# situation it can be slow, stochastic GD works by selecting a random instance and computes the gradient for that
# single observation. This means that SGD is quite 'bouncy' with each iteration. Another very importance benefit of
# this algo is the potential to avoid getting stuck in local minima and therefore find the global minima

# Data
np.random.seed(150)

X = 2 * np.random.rand(100, 1)
y = 20 + 7 * X + np.random.randn(100, 1) # y is 20 + 7 * X  + noise
X_beta = np.c_[np.ones((100, 1)), X]  # add the intercept of 1 to each observation
theta_best = np.linalg.inv(X_beta.T.dot(X_beta)).dot(X_beta.T).dot(y) # we compute the inverse of the matrix using the lin alg inv function, the matrix is the dot product

# Predictions
X_new = np.array([[0], [2]]) #
X_new_b = np.c_[np.ones((2,1)), X_new] # append to X_new

# params
n_epochs = 200 # iterations
t0, t1 = 10, 100  # parameters
m = len(X_beta) # rows in data

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # random initialization

for e in range(n_epochs):
    for i in range(m):
        y_predict = X_new_b.dot(theta)
        random_index = np.random.randint(m) # select obs
        xi = X_beta[random_index:random_index+1]  # select the beta
        yi = y[random_index:random_index+1] # select y
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi) # compute the gradient with T
        eta = learning_schedule(e * m + i) # eta
        theta = theta - eta * gradients # update

preds_SGD = X_new_b.dot(theta)

print('theta sgd: ', theta)
print('preds: ', preds_SGD)
