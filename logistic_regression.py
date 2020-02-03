"""
Logistic Regression
"""

#Author: Rancai Chen <chenrancai1993@gmail.com>


import numpy as np
import pandas as pd


class LogisticRegression:
    """
    Logistic regression based on gradient descent framework

    Parameters
    ----------
    lr: float, optional, default 0.01
        learning rate of model

    batch_size: flaot (0,1), int or None, default 1
        choose mini-batch to update gradient

    max_iter: int, optional, default 1000
        the iteration amount that the model will run

    multi_class: {'auto', 'binary', 'multinomial'}, optional, default 'auto'
        This parameter indicates the model to be trained either in binary or multinomial.
        By choosing 'auto', the model will recognize itself.

    keep_bias: bool, optional, default True
        keep intercept in linear parts or not

    reg_l1: float, optional, default 0.0
        l1 regularization

    reg_l2: float, optional, default 0.01
        l2 regularization

    shuffle: bool, optional, default False
        if shuffle the dataset before computing gradient

    Attributes
    ----------
    lr: float
        learning rate of gradient descent

    batch_size: flaot(0,1), int or None, default 1
        choose mini-batch to update gradient
        sgd: batch_szie=1

    max_iter: int
        maximum iteration times of updating coefficients

    multi_class: {'auto', 'binary', 'multinomial'}
        the type of classification model will build based on how many classes in the training dataset
        by choosing 'auto', the model  will recognize the type itself. The multi_class will be updated
        to either 'binary' or 'multinomial' after the training starts.

    keep_bias: bool
        if keep intercept in the linear parts

    reg_l1: float
        l1 regularization

    reg_l2: float
        l2 regularization

    shuffle: bool
        if shuffle the dataset before computing gradient

    activation: function
        the activation function that will be applied for training and prediction,
        it will be automatically assigned when the training starts.

    coef: dictionary
        the coefficients of linear parts, it will be initialized and updated
        when the training starts.

    """

    def __init__(self, lr=0.01, batch_size=1, max_iter=1000, multi_class='auto', keep_bias=True, reg_l1=0.0,
                 reg_l2=0.01,
                 shuffle=False):
        self.lr = lr
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.keep_bias = keep_bias
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.shuffle = shuffle
        self.activation = None
        self.coef = {'weights': None, 'bias': None}

    def initialize_coef(self, shape):
        """
        initialize the coefficients in the model

        Parameters
        ----------
        shape: tuple
            the shape of weights in the coefficients.
            i.e. for softmax: shape = (X.shape[1], y.shape[1])
        """

        # if the model is binary classification model, the weights will be initialized in different way
        if self.multi_class == 'binary':
            self.coef['weights'] = np.random.uniform(-0.00001, 0.00001, shape[0])
        else:
            self.coef['weights'] = np.random.uniform(-0.00001, 0.00001, shape)
        self.coef['bias'] = np.zeros(shape[1])

    def linear_function(self, X, w, b):
        """
        linear parts of model

        Parameters
        ----------
        X: numpy.ndarray
            2-D array features in the dataset

        w: numpy.ndarray
            1-D or 2-D array weights in coefficients

        b: numpy.ndarray
            1-D array bias in coefficients

        Returns
        -------
        numpy.ndarray
            return a 1-D or 2-D numpy array of linear expression: X*w + b.
            For binary classification, it will return 1-D array. For multinomial
            classification, it will return 2-D array
        """
        return (X.dot(w) + b)

    def sigmoid(self, z):
        """
        sigmoid activation function for binary classification

        Parameters
        ----------
        z: np.ndarray
            1-D numpy array

        Returns
        -------
        numpy.ndarray
            1-D numpy array which all the values are narrowed in range of (0, 1)
        """
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        """
        softmax activation function for multinomial classification

        Parameters
        ----------
        z: np.ndarray
            2-D numpy array

        Returns
        -------
        numpy.ndarray
            2-D numpy array which all the values are narrowed in range of (0, 1)
            and summation in direction of 1-axis equals one
        """
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def gradient(self, X, y, linear_func, act_func):
        """
        gradient computation

        Parameters
        ----------
        X: numpy.ndarray
            2-D features in the dataset

        y: numpy.ndarray
            1-D or 2-D targets in the dataset

        linear_func: function
            linear function to compute linear parts in the model

        act_func: function
            activation function to transform the linear results

        Returns
        -------
        grad_weights: numpy.ndarray
            the gradient to update weights in the coefficients

        grad_bias: numpy.ndarray
            the gradient to update bias in the coefficients
        """

        w = self.coef['weights']
        b = self.coef['bias']
        h = act_func(linear_func(X, w, b))
        grad_weights = X.T.dot(h - y) / y.shape[0]
        grad_bias = np.sum(h - y, axis=0) / y.shape[0]
        return grad_weights, grad_bias

    def update_coef(self, lr, grad_weights, grad_bias, l1, l2, keep_bias):
        """
        update coefficients  after receive gradients from gradient function

        Parameters
        ----------
        lr: float
            learning rate

        grad_weights: numpy.ndarray
            the gradient to update weights

        grad_bias: numpy.ndarray
            the gradient to update bias

        l1: float
            l1 regularization

        l2: float
            l2 regularization

        keep_bias: bool
            update bias or not
        """

        self.coef['weights'] -= lr * (grad_weights +
                                      l1 * (self.coef['weights'] / np.abs(self.coef['weights'])) +
                                      l2 * self.coef['weights'])
        if keep_bias:
            self.coef['bias'] -= lr * grad_bias

    def one_hot_encoder(self, y):
        return (np.arange(np.max(y) + 1) == y[:, None]).astype(float)

    def one_hot_to_label(self, pred):
        return np.argmax(pred, axis=1)

    def batch_generator(self, X, y, batch_size, shuffle):
        """
        initialize a mini-batch generator

        Parameters
        ----------
        X: numpy.ndarray
            2-D features in dataset

        y: numpy.ndarray
            1-D or 2-D targets in dataset
            it is a little different from the parameter in 'fit' method

        batch_size: float (0, 1) or int or None
            the batch_size to update weights and bias.
            if the batch_size input is a ratio expression, it will automatically convert it to
            real int batch_size.

        shuffle: bool
            if shuffle the dataset

        Yield
        -----
        numpy.ndarray
            mini-batch for gradient descent
        """

        if shuffle:
            shuffle_index = np.random.choice(X.shape[0], X.shape[0], replace=False)
            X = X[shuffle_index]
            y = y[shuffle_index]

        if not batch_size or batch_size < 0:
            batch_size = X.shape[0]
        if 0 < batch_size < 1:
            batch_size = int(X.shape[0] * batch_size)

        # when batch size greater than training data size, return training data
        if not X.shape[0] // batch_size:
            yield X, y

        else:
            for i in np.arange(0, X.shape[0], batch_size):
                x_train, y_train = X[i:i + batch_size], y[i:i + batch_size]
                yield x_train, y_train

    def is_binary_label(self, y):
        """check either is binary classification or multinomial classification"""

        unique = set()
        for i in range(y.shape[0]):
            if y[i] not in unique:
                unique.add(y[i])

            if len(unique) > 2:
                return False
        return True

    def pandas_to_numpy(self, data):
        """convert pandas to numpy type"""
        
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.to_numpy()

        elif isinstance(data, np.ndarray):
            return data

        else:
            raise TypeError("only accepts data in numpy or pandas type. Got %s." % (type(data)))

    def uniform_input(self, X, y=None):
        """convert X, y to numpy array form"""
        
        X = self.pandas_to_numpy(X)
        if X.ndim == 1:
            X = X.reshape((X.shape[0], -1))

        if not isinstance(y, (np.ndarray, pd.DataFrame, pd.Series)):
            return X

        y = self.pandas_to_numpy(y)
        if y.ndim == 2:
            y = y.reshape(y.shape[0])

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "inconsistant numbers of X and y. Got shape of X %s and shape of y %s." % (X.shape, y.shape))
        return X, y

    def fit(self, X, y, reset_coef=False):
        """
        train the model

        Parameters
        ----------
        X: numpy.ndarray
            1-D or 2-D array features in dataset

        y: numpy.ndarray
            1-D array in dataset, if it is not binary, this method will
            one-hot-encodes target into 2-D array.

        reset_coef: bool, defalut False
            if reset the coefficients or continute to train the model by using fit method.
            The result will be accumulated if choose False.

        Returns
        -------
        self
            return model itself
        """
        # convert X and y into uniform type
        X, y = self.uniform_input(X, y)

        # configure activation function and multi_class
        if not self.activation or not callable(self.activation):
            if self.multi_class == 'auto':
                if self.is_binary_label(y):
                    self.multi_class = 'binary'
                else:
                    self.multi_class = 'multinomial'

            if self.multi_class == 'binary':
                self.activation = self.sigmoid

            if self.multi_class == 'multinomial':
                self.activation = self.softmax

        # one hot encoding target if # of labels > 2
        if self.multi_class == 'multinomial':
            y = self.one_hot_encoder(y)

        # initialize regression coefficients
        if reset_coef or not isinstance(self.coef['weights'], np.ndarray):
            if self.multi_class == 'binary':
                self.initialize_coef((X.shape[1], 1))
            else:
                self.initialize_coef((X.shape[1], y.shape[1]))

        # train the model
        for i in range(self.max_iter):
            batches = self.batch_generator(X, y, self.batch_size, self.shuffle)
            for x_train, y_train in batches:
                grad_weights, grad_bias = self.gradient(x_train, y_train, self.linear_function, self.activation)
                self.update_coef(self.lr, grad_weights, grad_bias, self.reg_l1, self.reg_l2, self.keep_bias)
        return self

    def predict_proba(self, X):
        """
        predict results in probability form

        Parameters
        ----------
        X: numpy.ndarray
            2-D array features in the dataset

        Returns
        -------
        numpy.ndarray
            return the results in probabilities
        """

        X = self.uniform_input(X)

        prediction = self.activation(self.linear_function(X, self.coef['weights'], self.coef['bias']))
        return prediction

    def predict(self, X, threshold=0.5):
        """
        predict results and convert them into labels

        Parameters
        ----------
        X: numpy.ndarray
            features in dataset

        threshold: float (0, 1), default 0.5
            it only works for binary classification, the threshold
            will convert the probabilities into binary labels.

        Returns
        -------
        numpy.ndarray
            1-D array of label results of prediction
        """
        
        X = self.uniform_input(X)

        prediction = self.predict_proba(X)
        if prediction.ndim == 1:
            return np.array([1 if x >= threshold else 0 for x in prediction])
        else:
            return self.one_hot_to_label(prediction)
