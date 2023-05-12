import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def plot_residuals(y, yhat):
    residuals = y - yhat
    baseline = y.mean()
    # baseline
    plt.axhline(baseline, ls=':', color='black')
    sns.scatterplot(y = residuals, x = y, hue = y)
    #plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()



def regression_errors(y, yhat):
    errors = y - yhat
    squared_errors = errors ** 2
    
    SSE = np.sum(squared_errors)
    ESS = np.sum((yhat - np.mean(y)) ** 2)
    TSS = np.sum((y - np.mean(y)) ** 2)
    MSE = np.mean(squared_errors)
    RMSE = np.sqrt(MSE)
    
    return SSE, ESS, TSS, MSE, RMSE


def baseline_mean_errors(y):
    baseline_prediction = np.mean(y)
    errors = y - baseline_prediction
    squared_errors = errors ** 2
    
    SSE = np.sum(squared_errors)
    MSE = np.mean(squared_errors)
    RMSE = np.sqrt(MSE)
    
    return SSE, MSE, RMSE

def better_than_baseline(y, yhat, baseline):
    sse_model = np.sum((y - yhat) ** 2)
    sse_baseline = np.sum((y - baseline) ** 2)

    return sse_model < sse_baseline

#recursive feature elimination
def rfe(X, y, k):
    estimator = LinearRegression()
    selector = RFE(estimator, n_features_to_select=k)
    selector.fit(X, y)
    mask = selector.support_
    selected_features = X.columns[mask]
    return selected_features

#K best feature selection
def select_kbest(X, y, k):
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    mask = selector.get_support()
    selected_features = X.columns[mask]
    return selected_features

def run_regression1(df, target_var):
    columns = df.columns.tolist()
    combinations = [combo for r in range(1, len(columns) + 1) for combo in itertools.combinations(columns, r)]
    best_rmse = float('inf')
    best_rmse_combo = None
    best_r2 = float('-inf')
    best_r2_combo = None

    for combo in combinations:
        X = df[list(combo)]
        y = target_var

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reg = LinearRegression().fit(X_scaled, y)
        y_pred = reg.predict(X_scaled)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        if rmse < best_rmse:
            best_rmse = rmse
            best_rmse_combo = combo

        if r2 > best_r2:
            best_r2 = r2
            best_r2_combo = combo

        print(f'RMSE: {rmse:.2f}, R^2: {r2:.2f} for {combo}')

    sns.regplot(x=y_pred, y=y, line_kws={'color':'red'}, scatter_kws={'alpha':0.06})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print(f'Best RMSE: {best_rmse:.2f} for {best_rmse_combo}')
    print(f'Best R^2: {best_r2:.2f} for {best_r2_combo}')
    return best_r2_combo

def metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

