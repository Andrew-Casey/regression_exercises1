from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def random_forest_scores(X_train, y_train, X_validate, y_validate):
    """
    Trains and evaluates a random forest classifier with different combinations of hyperparameters. The function takes in 
    training and validation datasets, and returns a dataframe summarizing the model performance on each combination of 
    hyperparameters.

    Parameters:
    -----------
    X_train : pandas DataFrame
        Features of the training dataset.
    y_train : pandas Series
        Target variable of the training dataset.
    X_validate : pandas DataFrame
        Features of the validation dataset.
    y_validate : pandas Series
        Target variable of the validation dataset.

    Returns:
    --------
    df : pandas DataFrame
        A dataframe summarizing the model performance on each combination of hyperparameters.
    """
    #define variables
    train_scores = []
    validate_scores = []
    min_samples_leaf_values = [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10]
    max_depth_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    
    for min_samples_leaf, max_depth in zip(min_samples_leaf_values, max_depth_values):
        rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth,random_state=123)
        rf.fit(X_train, y_train)
        train_score = rf.score(X_train, y_train)
        validate_score = rf.score(X_validate, y_validate)
        train_scores.append(train_score)
        validate_scores.append(validate_score)
       
    # Calculate the difference between the train and validation scores
    diff_scores = [train_score - validate_score for train_score, validate_score in zip(train_scores, validate_scores)]
    
    #Put results into a dataframe
    df = pd.DataFrame({
        'min_samples_leaf': min_samples_leaf_values,
        'max_depth': max_depth_values,
        'train_score': train_scores,
        'validate_score': validate_scores,
        'diff_score': diff_scores})
     
    # Set plot style
    sns.set_style('whitegrid')
 
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(max_depth_values, train_scores, label='train', marker='o', color='blue')
    plt.plot(max_depth_values, validate_scores, label='validation', marker='o', color='orange')
    plt.fill_between(max_depth_values, train_scores, validate_scores, alpha=0.2, color='gray')
    plt.xticks([2,4,6,8,10],['Leaf 9 and Depth 2','Leaf 7 and Depth 4','Leaf 5 and Depth 6','Leaf 3 and Depth 8','Leaf 1and Depth 10'], rotation = 45)
    plt.xlabel('min_samples_leaf and max_depth', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Random Forest Classifier Performance', fontsize=18)
    plt.legend(fontsize=12)
    plt.show()
    
    return df


def plot_logistic_regression(X_train, X_validate, y_train, y_validate):
    '''
    Trains multiple logistic regression models with different regularization strengths (C) on the given training
    data, and plots the resulting train and validation scores against C values. The optimal value of C is marked
    by a vertical red dashed line, and the associated difference between the train and validation scores is shown
    in the plot legend.

    Parameters:
    X_train : array-like of shape (n_samples, n_features)
        The training input samples.
    X_validate : array-like of shape (n_samples, n_features)
        The validation input samples.
    y_train : array-like of shape (n_samples,)
        The target values for training.
    y_validate : array-like of shape (n_samples,)
        The target values for validation.

    Returns:
    df1 : pandas DataFrame
        A table containing the C, train_score, validate_score, and diff_score values for each model.
    '''
    train_scores = []
    val_scores = []
    c_values = [.01, .1, 1, 10 , 100, 1000]
    for c in c_values:
        logit = LogisticRegression(C=c, random_state=123)
        logit.fit(X_train, y_train)
        train_score = logit.score(X_train, y_train)
        val_score = logit.score(X_validate, y_validate)
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    # Calculate the difference between the train and validation scores
    diff_scores = [train_score - val_score for train_score, val_score in zip(train_scores, val_scores)]
     
    # Put results into a list of tuples
    results = list(zip(c_values, train_scores, val_scores, diff_scores))
    # Convert the list of tuples to a Pandas DataFrame
    df1 = pd.DataFrame(results, columns=['C', 'train_score', 'validate_score', 'diff_score'])
    

    # Plot the results
    plt.plot(c_values, train_scores, label='Train score')
    plt.plot(c_values, val_scores, label='Validation score')
    min_diff_idx = np.abs(diff_scores).argmin()
    min_diff_c = results[min_diff_idx][0]
    min_diff_score = results[min_diff_idx][3]
    plt.axvline(min_diff_c, linestyle='--', linewidth=2, color='red', label=f'min diff at C={min_diff_c} (diff={min_diff_score:.3f})')
    plt.fill_between(c_values, train_scores, val_scores, alpha=0.2, color='gray')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regression Accuracy vs C')
    plt.legend()
    plt.show()

    return df1

