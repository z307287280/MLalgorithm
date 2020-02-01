import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, max_iter=1000, multi_class='auto', keep_bias=True, reg_l1=0.0, reg_l2=0.01,
                 shuffle=False):
        self.lr = lr
        self.max_iter = max_iter
        self.keep_bias = keep_bias
        self.shuffle = shuffle
        self.multi_class = multi_class
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.activation = None
        self.coef = {'weights': None, 'bias': None}

    def initialize_coef(self, shape):
        if self.multi_class == 'binary':
            self.coef['weights'] = np.random.uniform(-0.00001, 0.00001, shape[0])
        else:
            self.coef['weights'] = np.random.uniform(-0.00001, 0.00001, shape)
        self.coef['bias'] = np.zeros(shape[1])

    def linear_function(self, X, w, b):
        return (X.dot(w) + b)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def gradient(self, X, y, linear_func, act_func):
        w = self.coef['weights']
        b = self.coef['bias']
        h = act_func(linear_func(X, w, b))

        grad_weights = X.T.dot(h - y) / y.shape[0]
        grad_bias = np.sum(h - y, axis=0) / y.shape[0]
        return grad_weights, grad_bias

    def update_coef(self, lr, grad_weights, grad_bias, l1, l2, keep_bias):
        self.coef['weights'] -= lr * (grad_weights + \
                                      l1 * (self.coef['weights'] / np.abs(self.coef['weights'])) + \
                                      l2 * self.coef['weights'])
        if keep_bias:
            self.coef['bias'] -= lr * grad_bias

    def one_hot_encoder(self, y):
        return (np.arange(np.max(y) + 1) == y[:, None]).astype(float)

    def one_hot_to_label(self, pred):
        return np.argmax(pred, axis=1)

    def batch_generator(self, X, y, batch_size, shuffle):
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
        unique = set()
        for i in range(y.shape[0]):
            if y[i] not in unique:
                unique.add(y[i])
            if len(unique) > 2:
                return False
        return True

    def fit(self, X, y, batch_size=None, reset_coef=False):
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

        for i in range(self.max_iter):
            batches = self.batch_generator(X, y, batch_size, self.shuffle)
            for x_train, y_train in batches:
                grad_weights, grad_bias = self.gradient(x_train, y_train, self.linear_function, self.activation)
                self.update_coef(self.lr, grad_weights, grad_bias, self.reg_l1, self.reg_l2, self.keep_bias)

    def predict_proba(self, X):
        prediction = self.activation(self.linear_function(X, self.coef['weights'], self.coef['bias']))
        return prediction

    def predict(self, X, threshold=0.5):
        prediction = self.predict_proba(X)
        if prediction.ndim == 1:
            return np.array([1 if x >= threshold else 0 for x in prediction])
        else:
            return self.one_hot_to_label(prediction)
