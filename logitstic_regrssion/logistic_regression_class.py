import numpy as np
import sklearn.datasets

class LogisticRegression:

    def __init__(self, lr=0.01, num_iter=10000, fit_intercept=True, verbose = False):
        '''
        Initialization of parameters
        '''
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose


    def __add_intercept(self, X):
        '''
        Add additional column in X (additional features)
        '''
        #create a vector of parameters
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)


    def __sigmoid(self, z):
        '''
        Sigmoid
        '''
        return 1 / (1 + np.exp(-z))


    def __loss(self, h, y):
        '''
        Loss function
        '''
        return (-y * np.log(h), - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold = 0.5):
        return self.predict_prob(X) >= threshold

def main():
    model = LogisticRegression(lr = 0.1, num_iter=1000)
    iris = sklearn.datasets.load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1
    model.fit(X, y)
    print(model.theta)

if __name__ == "__main__":
    main()
    