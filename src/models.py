import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels



class RegressionLineaire(BaseEstimator, ClassifierMixin):

    def __init__(self, max_iter=1000, eps=1e-1):
        self.eps = eps
        self.max_iter = max_iter

    def fit(self, X, y):

        # adding the bias to the X data
        X = np.hstack((X, np.ones((len(X), 1))))
        _, n = X.shape


        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        self.w = np.random.random(n)
        self.__fit_batch()

        # Return the classifier
        return self

    def __fit_batch(self):
        self.training_errors = np.zeros(self.max_iter)
        for i in range(self.max_iter):
            J, grad = self.__mse(self.X_, self.y_)
            self.w = self.w - self.eps * grad
            self.training_errors[i] = J

    def __mse(self, datax, datay):
        """ retourne la moyenne de l'erreur aux moindres carres """
        m, _ = datax.shape

        if len(datax.shape) == 1:
            datax = datax.reshape(1, -1)

        pred = np.dot(datax, self.w)

        J = (1/(2*m)) * ((pred-datay)**2).sum()
        grad = (1/m)*np.dot(datax.T, pred-datay)

        return J, grad

    def predict(self, X):

        X = np.hstack((X, np.ones((len(X), 1))))

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return np.sign(np.dot(X, self.w.reshape(-1)))


class RegressionRIDGE(BaseEstimator, ClassifierMixin):

    def __init__(self, max_iter=1000, eps=1e-1,alpha=0.1):
        self.alpha = alpha
        self.eps = eps
        self.max_iter = max_iter

    def fit(self, X, y):

        # adding the bias to the X data
        X = np.hstack((X, np.ones((len(X), 1))))
        _, n = X.shape


        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        self.w = np.random.random(n)
        self.__fit_batch()

        # Return the classifier
        return self

    def __fit_batch(self):
        self.training_errors = np.zeros(self.max_iter)
        for i in range(self.max_iter):
            J, grad = self.__mse(self.X_, self.y_)
            self.w = self.w - self.eps * grad
            self.training_errors[i] = J

    def __mse(self, datax, datay):
        """ retourne la moyenne de l'erreur aux moindres carres """
        m, _ = datax.shape

        if len(datax.shape) == 1:
            datax = datax.reshape(1, -1)

        pred = np.dot(datax, self.w)

        J = (1/(2*m)) * ((pred-datay)**2).sum()  + self.alpha * (self.w**2).sum()
        grad = (1/m)*np.dot(datax.T, pred-datay) + self.alpha * (self.w)

        return J, grad

    def predict(self, X):

        X = np.hstack((X, np.ones((len(X), 1))))

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return np.sign(np.dot(X, self.w.reshape(-1)))

class RegressionLASSO(BaseEstimator, ClassifierMixin):

    def __init__(self, max_iter=1000, eps=1e-1,alpha=0.1):
        self.alpha = alpha
        self.eps = eps
        self.max_iter = max_iter

    def fit(self, X, y):

        # adding the bias to the X data
        X = np.hstack((X, np.ones((len(X), 1))))
        _, n = X.shape


        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        self.w = np.random.random(n)
        self.__fit_batch()

        # Return the classifier
        return self

    def __fit_batch(self):
        self.training_errors = np.zeros(self.max_iter)
        for i in range(self.max_iter):
            J, grad = self.__mse(self.X_, self.y_)
            self.w = self.w - self.eps * grad
            self.training_errors[i] = J

    def __mse(self, datax, datay):
        """ retourne la moyenne de l'erreur aux moindres carres """
        m, _ = datax.shape

        if len(datax.shape) == 1:
            datax = datax.reshape(1, -1)

        pred = np.dot(datax, self.w)

        J = (1/(2*m)) * ((pred-datay)**2).sum()  + self.alpha * (self.w**2).sum()
        grad = (1/m)*np.dot(datax.T, pred-datay) + self.alpha * (self.w)

        return J, grad

    def predict(self, X):

        X = np.hstack((X, np.ones((len(X), 1))))

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return np.sign(np.dot(X, self.w.reshape(-1)))
