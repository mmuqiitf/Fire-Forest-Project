import numpy as np
# X and F are numpy matrices
def convolve(X, F):
    X_height = X.shape[0]
    X_width = X.shape[1]
    F_height = F.shape[0]
    F_width = F.shape[1]
    H = F_height // 2
    W = F_width // 2
    out = np.zeros((X_height, X_width))
    for i in np.arange(H+1, X_height - H):
        for j in np.arange(W+1, X_width - W):
            total = 0
            for k in np.arange(-H, H + 1):
                for l in np.arange(-W, W + 1):
                    a = X[i + k, j + l]
                    w = F[H + k, W + l]
                    total += (w * a)
            out[i, j] = total
    return out
