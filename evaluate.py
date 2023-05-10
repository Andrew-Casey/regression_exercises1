import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


