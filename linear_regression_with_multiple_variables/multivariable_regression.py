import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 5, 15], [1, 4, 20], [1, 2, 18]])
Y = np.array([4,5,6], ndmin=2).T
params = np.zeros((1, 3)).T
learning_rate = 0.01

print(X)
print(Y)

print()


predict = np.dot(X, params)
error = (predict - Y) ** 2
print("error befor:", error)
print("params befor:", params)

epochs = 1000

for i in range(epochs):
    grad_vector = np.zeros((1,3)).T
    for x,y in zip(X,np.reshape(Y,-1)):
        grad_vector += (np.dot(x,params) - y)
    params -= learning_rate * grad_vector


predict = np.dot(X, params)
error = (predict - Y) ** 2
print()
print("error after:", error)
print("params after:", params)



