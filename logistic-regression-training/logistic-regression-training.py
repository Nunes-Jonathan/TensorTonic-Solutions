import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""

    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)

    pos = z >= 0
    neg = ~pos

    out[pos] = 1 / (1 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1 + exp_z)
    
    return out

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    N,D = X.shape
    w = np.zeros(D, dtype=float)
    b = 0.0

    losses = []

    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)

        p_clip = np.clip(p, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))
        losses.append(loss)

        error = p - y

        dw = (X.T @ error) / N
        db = np.mean(error)

        w -= lr * dw
        b -= lr * db

    return w, b