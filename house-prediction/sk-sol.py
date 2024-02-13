from sklearn import linear_model
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import mean_squared_error, r2_score

def main():
    """
    A general structure for this hand-written kernel for a simple Linear
    Regression on Housing Price Dataset will be as follows:
        1. Load and clean the data     (done)
        2. Implement compute_cost      (done)
        3. Implement compute_grad      (done)
        4. Implement Gradient Descent
        5. Test it
        6. Implement Vectorization
        7. Implement Regularisation

    """
    raw_data = pd.read_csv("train.csv")

    X = raw_data[["LotArea", "OverallQual", "OverallCond", "YearBuilt",
                        "YrSold", "PoolArea", "ScreenPorch", "GarageCars",
                        "Fireplaces", "TotRmsAbvGrd"]].to_numpy()
    m, n = X.shape
    y = raw_data["SalePrice"].to_numpy().flatten()

    print(f"X: {X}\n y: {y}")
    
    regr = linear_model.LinearRegression()
    
    
    regr.fit(X, y)
    y_pred = regr.predict(X)
    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y, y_pred))
    
main()