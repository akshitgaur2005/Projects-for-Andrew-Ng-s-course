import pandas as pd
import numpy as np
import copy

def main():
    raw_data = pd.read_csv("train.csv")

    X = raw_data[["LotArea", "OverallQual", "OverallCond", "YearBuilt",
                  "YrSold", "PoolArea", "ScreenPorch", "GarageCars",
                  "Fireplaces", "TotRmsAbvGrd"]].to_numpy()
    m, n = X.shape
    y = raw_data["SalePrice"].to_numpy().flatten()

    # Normalize features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    w = np.random.rand(n)  # Initialize weights to random values
    b = 0.0  # Initialize bias

    w_new, b_new, J = gradient_descent(X, y, w, b, 0.00001, 10e-4, 1500000)

    print(f"J: \n{J[-1]}")
    print(f"w: {w_new}\nb: {b_new}")

def compute_cost(X, y, w, b, lmb):
    m = len(y)
    f_wb = np.dot(X, w) + b
    diff = f_wb - y
    cost = (np.sum(diff ** 2) + lmb * np.sum(w**2)) / (2 * m)
    return cost

def compute_grad(X, y, w, b, lmb):
    m = len(y)
    f_wb = np.dot(X, w) + b
    diff = f_wb - y
    dj_dw = (np.dot(diff, X) + lmb * w) / m
    dj_db = np.sum(diff) / m
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, lr, lmb, num_iters):
    m = len(y)
    w_in = copy.deepcopy(w)
    b_in = b
    J = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_grad(X, y, w_in, b_in, lmb)
        w_in = w_in - lr * dj_dw
        b_in = b_in - lr * dj_db
        j = compute_cost(X, y, w_in, b_in, lmb)
        J.append(j)

        if (i % 10) == 0:
            print(f"i: {i}\tJ: {j}")

    return w_in, b_in, J

main()
