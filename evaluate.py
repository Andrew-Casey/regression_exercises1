import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression

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

import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def run_regression(df, target_var):
    columns = df.columns.tolist()
    combinations = [combo for r in range(1, len(columns) + 1) for combo in itertools.combinations(columns, r)]
    best_rmse = float('inf')
    best_rmse_combo = None
    best_r2 = float('-inf')
    best_r2_combo = None

    for combo in combinations:
        X = df[list(combo)]
        y = target_var
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        if rmse < best_rmse:
            best_rmse = rmse
            best_rmse_combo = combo

        if r2 > best_r2:
            best_r2 = r2
            best_r2_combo = combo

        print(f'RMSE: {rmse:.2f}, R^2: {r2:.2f} for {combo}')

    plt.scatter(y, y_pred, alpha=0.2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()

    print(f'Best RMSE: {best_rmse:.2f} for {best_rmse_combo}')
    print(f'Best R^2: {best_r2:.2f} for {best_r2_combo}')
    return best_r2_combo




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
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        if rmse < best_rmse:
            best_rmse = rmse
            best_rmse_combo = combo

        if r2 > best_r2:
            best_r2 = r2
            best_r2_combo = combo

        print(f'RMSE: {rmse:.2f}, R^2: {r2:.2f} for {combo}')

    # Select the best combination of features
    X_best = df[list(best_r2_combo)]
    y_best = target_var

    # Fit the linear regression model using the best combination of features
    reg_best = LinearRegression().fit(X_best, y_best)

    # Calculate the predicted values using the best model
    y_pred_best = reg_best.predict(X_best)

    # Sample the data
    sample_size = 5000
    df_sampled = df.sample(sample_size).reset_index(drop=True)
    y_pred_sampled = y_pred_best[df_sampled.index]

    # Create a new dataframe with actual and predicted values
    df_predicted = pd.DataFrame({
        'Actual': target_var[df_sampled.index],
        'Predicted': y_pred_sampled
    })


    # Create scatter plot with regression line and hue based on 'taxvalue'
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_predicted, x='Actual', y='Predicted', alpha=0.2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()

    print(f'Best RMSE: {best_rmse:.2f} for {best_rmse_combo}')
    print(f'Best R^2: {best_r2:.2f} for {best_r2_combo}')
    return best_r2_combo

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import itertools

def run_regression2(df, target_var):
    y_train_reset = target_var.reset_index(drop=True)  # Reset index of target variable

    columns = df.columns.tolist()
    combinations = [combo for r in range(1, len(columns) + 1) for combo in itertools.combinations(columns, r)]
    best_rmse = float('inf')
    best_rmse_combo = None
    best_r2 = float('-inf')
    best_r2_combo = None

    for combo in combinations:
        X = df[list(combo)]
        y = y_train_reset  # Use the reset target variable
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        if rmse < best_rmse:
            best_rmse = rmse
            best_rmse_combo = combo

        if r2 > best_r2:
            best_r2 = r2
            best_r2_combo = combo

        print(f'RMSE: {rmse:.2f}, R^2: {r2:.2f} for {combo}')

    # Select the best combination of features
    X_best = df[list(best_r2_combo)]
    y_best = y_train_reset  # Use the reset target variable

    # Fit the linear regression model using the best combination of features
    reg_best = LinearRegression().fit(X_best, y_best)

    # Calculate the predicted values using the best model
    y_pred_best = reg_best.predict(X_best)

    # Sample the data
    sample_size = 10000
    df_sampled = df.sample(sample_size).reset_index(drop=True)
    y_pred_sampled = y_pred_best[df_sampled.index]

    # Create a new dataframe with actual and predicted values
    df_predicted = pd.DataFrame({
        'Actual': target_var[df_sampled.index],
        'Predicted': y_pred_sampled
    })

    # Create scatter plot with regression line and hue based on 'taxvalue'
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_predicted, x='Actual', y='Predicted', alpha=0.2)
    
    # Add regression line
    sns.regplot(data=df_predicted, x='Actual', y='Predicted', scatter=False, color='red')
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()

    print(f'Best RMSE: {best_rmse:.2f} for {best_rmse_combo}')
    print(f'Best R^2: {best_r2:.2f} for {best_r2_combo}')
    return best_r2_combo


def rfe(X, y, k):
    estimator = LinearRegression()
    selector = RFE(estimator, n_features_to_select=k)
    selector.fit(X, y)
    mask = selector.support_
    selected_features = X.columns[mask]
    return selected_features

def select_kbest(X, y, k):
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    mask = selector.get_support()
    selected_features = X.columns[mask]
    return selected_features




