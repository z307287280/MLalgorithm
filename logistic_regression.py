import numpy as np


class LogisticRegression:
    """
    Logistic regression based on gradient descent framework

    Parameters
    __________
    lr: float, optional, default 0.01
        learning rate of model

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
    __________
    lr: float
        learning rate of gradient descent

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
    def __init__(self, lr=0.01, max_iter=1000, multi_class='auto', keep_bias=True, reg_l1=0.0, reg_l2=0.01,
                 shuffle=False):
        self.lr = lr
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
        __________
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
        __________
        X: numpy.ndarray
            features in the dataset

        w: numpy.ndarray
            weights in coefficients

        b: numpy.ndarray
            bias in coefficients

        Returns
        _______
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
        __________
        z: np.ndarray
            1-D numpy array

        Returns
        _______
        numpy.ndarray
            1-D numpy array which all the values are narrowed in range of (0, 1)
        """
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        """
        softmax activation function for multinomial classification

        Parameters
        __________
        z: np.ndarray
            2-D numpy array

        Returns
        _______
        numpy.ndarray
            2-D numpy array which all the values are narrowed in range of (0, 1)
            and summation in direction of 1-axis equals one
        """
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def gradient(self, X, y, linear_func, act_func):
        """
        gradient computation

        Parameters
        __________
        X: numpy.ndarray
            features in the dataset

        y: numpy.ndarray
            targets in the dataset

        linear_func: function
            linear function to compute linear parts in the model

        act_func: function
            activation function to transform the linear results

        Returns
        _______
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
        __________
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
        self.coef['weights'] -= lr * (
                    grad_weights + l1 * (self.coef['weights'] / np.abs(self.coef['weights'])) + l2 * self.coef[
                'weights'])
        if keep_bias:
            self.coef['bias'] -= lr * grad_bias

    def one_hot_encoder(self, y):
        return (np.arange(np.max(y) + 1) == y[:, None]).astype(float)

    def one_hot_to_label(self, pred):
        return np.argmax(pred, axis=1)

    def batch_generator(self, X, y, batch_size, shuffle):
        """
        generate a mini-batch generator

        Parameters
        __________
        X: numpy.ndarray
            features in dataset

        y: numpy.ndarray
            targets in dataset

        shuffle: bool
            if shuffle the dataset

        Yield
        _____
        numpy.ndarray
            mini-batch for gradient descent
        """

        if self.multi_class == 'binary':
            y = y.reshape((-1, 1))

        dataset = np.hstack((X, y))
        index = X.shape[1]

        if shuffle:
            np.random.shuffle(dataset)
        if not batch_size or batch_size < 0:
            batch_size = dataset.shape[0]
        if 0 < batch_size <= 1:
            batch_size = int(dataset.shape[0] * batch_size)

        num_batches = dataset.shape[0] // batch_size
        if dataset.shape[0] % batch_size:
            num_batches += 1

        for i in range(num_batches):
            x_train = dataset[i * batch_size:(i + 1) * batch_size, :index]
            if self.multi_class == 'binary':
                y_train = dataset[i * batch_size:(i + 1) * batch_size, -1]
            else:
                y_train = dataset[i * batch_size:(i + 1) * batch_size, index:]
            yield x_train, y_train

    def is_binary_label(self, y):
        # check either is binary classification or multinomial classification

        unique = set()
        for i in range(y.shape[0]):
            if y[i] not in unique:
                unique.add(y[i])
            if len(unique) > 2:
                return False
        return True

    def fit(self, X, y, batch_size=None, reset_coef=False):
        """
        train the model

        Parameters
        __________
        X: numpy.ndarray
            features in dataset

        y: numpy.ndarray
            target in dataset, if it is not binary, this method will
            one-hot-encodes y into 2-D array.

        batch_size: int or None, default None
            choose mini-batch to update gradient

        reset_coef: bool, defalut False
            if reset the coefficients or continute to train the model by using fit method.
            The result will be accumulated if choose False.

        Returns
        _______
        self
            return model itself
        """

        # configure activation function
        if not self.activation:
            if self.multi_class == 'auto':
                if self.is_binary_label(y):
                    self.multi_class = 'binary'
                    self.activation = self.sigmoid
                else:
                    self.multi_class = 'multinomial'
                    self.activation = self.softmax

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
            batches = self.batch_generator(X, y, batch_size, self.shuffle)
            for x_train, y_train in batches:
                grad_weights, grad_bias = self.gradient(x_train, y_train, self.linear_function, self.activation)
                self.update_coef(self.lr, grad_weights, grad_bias, self.reg_l1, self.reg_l2, self.keep_bias)
        return self

    def predict_proba(self, X):
        """
        predict results in probability form

        Parameters
        __________
        X: numpy.ndarray
            features in the dataset

        Returns
        _______
        numpy.ndarray
            return the results in probabilities
        """
        prediction = self.activation(self.linear_function(X, self.coef['weights'], self.coef['bias']))
        return prediction

    def predict(self, X, threshold=0.5):
        """
        predict results and convert them into labels

        Parameters
        __________
        X: numpy.ndarray
            features in dataset

        threshold: float (0, 1), default 0.5
            it only works for binary classification, the threshold
            will convert the probabilities into binary labels.

        Returns
        _______
        numpy.ndarray
            1-D array of label results of prediction
        """
        prediction = self.predict_proba(X)
        if prediction.ndim == 1:
            return np.array([1 if x >= threshold else 0 for x in prediction])
        else:
            return self.one_hot_to_label(prediction)