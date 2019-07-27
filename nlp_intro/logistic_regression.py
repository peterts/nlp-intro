import numpy as np


def logistic_func(w, X):
    return 1 / (1 + np.exp(-net(w, X)))


def net(w, X):
    return np.sum(w * X, axis=1, keepdims=True)


def gradient(w, X, y):
    y_pred = logistic_func(w, X)
    return np.mean((y_pred - y) * y_pred * (1 - y_pred) * X, axis=0)


def update_weights(w, X, y, lr):
    return w - lr * gradient(w, X, y)


def fit(X, y, lr=.1, fit_intercept=True, w=None, max_iter=100, tol=1e-3, verbose=True):
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.expand_dims(X, axis=1)
    y = np.expand_dims(y, axis=1)
    if w is not None:
        w = np.asarray(w)

    # Add intercept to data
    if fit_intercept:
        const = np.ones((X.shape[0], 1))
        X = np.hstack([X, const])

    # Initialize weights
    if w is None:
        # Randomly select a set of starting weights between -1 and 1
        w = -1 + 2 * np.random.random(X.shape[1])
    assert w.shape[0] == X.shape[1]

    # Fit weights using gradient descent
    for i in range(max_iter):
        grad = gradient(w, X, y)
        if verbose:
            print(f"Weights: {w}, Gradient: {grad}")
        w = w - lr * grad

    return w
