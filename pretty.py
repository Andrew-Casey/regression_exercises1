import pro_acquire as acq
import pro_prepare as prep
import pro_model as pm
import knear as k

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_that_data():
    """
    This function prepares data from the 'telco' dataset for modeling by dropping unnecessary columns, moving the target
    variable 'churn_encoded' to index level 1 of the dataframe, and splitting the data into training, validation, and test
    sets using the 'split_data()' function from the 'prep' module.

    Returns:
    - train (pandas.DataFrame): A DataFrame containing the training data.
    - validate (pandas.DataFrame): A DataFrame containing the validation data.
    - test (pandas.DataFrame): A DataFrame containing the test data.
    """
    #get data
    df = prep.prep_telco()

    #drop uncessary columns
    df = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id','gender',
                           'senior_citizen','partner','dependents','phone_service','multiple_lines','online_security',
                           'online_backup','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing',
                           'churn', 'contract_type','internet_service_type','payment_type'])

    #move target (churn) to index level 1 of dataframe
    column_to_move = df.pop("churn_encoded")
    df.insert(1, "churn_encoded", column_to_move)
    
    # split data using function
    train, validate, test = prep.split_data(df, 'churn_encoded')
    
    return train, validate, test

def modeling(train, validate, test):
    """
    This function sets the features and target variables for modeling from the provided training, validation, and test
    datasets. The features selected for modeling are customer_id, tenure, monthly_charges, senior_citizen_encoded, and
    tech_support_Yes. The target variable is 'churn_encoded'.

    Parameters:
    - train (pandas.DataFrame): A DataFrame containing the training data.
    - validate (pandas.DataFrame): A DataFrame containing the validation data.
    - test (pandas.DataFrame): A DataFrame containing the test data.

    Returns:
    - X_train (pandas.DataFrame): A DataFrame containing the features for the training data.
    - X_validate (pandas.DataFrame): A DataFrame containing the features for the validation data.
    - X_test (pandas.DataFrame): A DataFrame containing the features for the test data.
    - y_train (pandas.Series): A Series containing the target variable for the training data.
    - y_validate (pandas.Series): A Series containing the target variable for the validation data.
    - y_test (pandas.Series): A Series containing the target variable for the test data.
    """
    #set features
    xt = train[['customer_id','tenure','monthly_charges','senior_citizen_encoded', 'tech_support_Yes']]
    xv = validate[['customer_id','tenure','monthly_charges','senior_citizen_encoded', 'tech_support_Yes']]
    xtest = test[['customer_id','tenure','monthly_charges','senior_citizen_encoded', 'tech_support_Yes']]
    # set variables for modeling
    X_train = xt.iloc[:,1:]
    X_validate = xv.iloc[:,1:]
    X_test = xtest.iloc[:,1:]
    #set target
    target = 'churn_encoded'
    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]

    return X_train, X_validate, X_test, y_train, y_validate, y_test


def decision_tree(X_train, X_validate, y_train, y_validate):
    """
    This function trains a decision tree classifier on the provided training data, and evaluates its performance on the
    validation data for different values of the 'max_depth' hyperparameter. It then generates a plot of the training and
    validation accuracy scores as a function of 'max_depth', and returns a DataFrame containing these scores.

    Parameters:
    - X_train (pandas.DataFrame): A DataFrame containing the features for the training data.
    - X_validate (pandas.DataFrame): A DataFrame containing the features for the validation data.
    - y_train (pandas.Series): A Series containing the target variable for the training data.
    - y_validate (pandas.Series): A Series containing the target variable for the validation data.

    Returns:
    - scores_df (pandas.DataFrame): A DataFrame containing the training and validation accuracy scores, as well as the
      difference between them, for different values of the 'max_depth' hyperparameter.
    """
    # get data
    scores_all = []
    for x in range(1,20):
        tree = DecisionTreeClassifier(max_depth=x, random_state=123)
    
        tree.fit(X_train, y_train)
        train_acc = tree.score(X_train,y_train)
        val_acc = tree.score(X_validate, y_validate)
        score_diff = train_acc - val_acc
        scores_all.append([x, train_acc, val_acc, score_diff])
    
    scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc','val_acc','score_diff'])
    
    # Plot the results
    sns.set_style('whitegrid')
    plt.plot(scores_df['max_depth'], scores_df['train_acc'], label='Train score')
    plt.plot(scores_df['max_depth'], scores_df['val_acc'], label='Validation score')
    plt.fill_between(scores_df['max_depth'], scores_df['train_acc'], scores_df['val_acc'], alpha=0.2, color='gray')
    plt.xlabel('Max depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Max Depth')
    plt.legend()
    plt.show()

    return scores_df



def the_chosen_one(X_train, X_validate, X_test, y_train, y_validate, y_test):
    """
    This function trains a random forest classifier on the provided training data, with hyperparameters chosen based on
    previous experimentation. It then evaluates the classifier on the test data and returns the test accuracy score and
    the trained classifier object.

    Parameters:
    - X_train (pandas.DataFrame): A DataFrame containing the features for the training data.
    - X_validate (pandas.DataFrame): A DataFrame containing the features for the validation data.
    - X_test (pandas.DataFrame): A DataFrame containing the features for the test data.
    - y_train (pandas.Series): A Series containing the target variable for the training data.
    - y_validate (pandas.Series): A Series containing the target variable for the validation data.
    - y_test (pandas.Series): A Series containing the target variable for the test data.

    Returns:
    - test_acc (float): The accuracy score of the trained random forest classifier on the test data.
    - rf1 (RandomForestClassifier object): The trained random forest classifier object.
    """

    rf1 = RandomForestClassifier(max_depth=6, min_samples_leaf=5, random_state = 123)
    rf1.fit(X_train, y_train)
    rf1.score(X_train, y_train)
    rf1.score(X_validate, y_validate)
    rf1.score(X_test, y_test)

    return rf1.score(X_test, y_test), rf1

def make_that_csv(X_test, test, rf1):
    """
    Export the predictions and prediction probabilities of a given test set into a CSV file.

    Parameters:
    -----------
    X_test: pandas DataFrame
        The feature matrix of the test set.
    test: pandas DataFrame
        The original test set.
    rf1: RandomForestClassifier object
        The fitted random forest classifier.

    Returns:
    --------
    None
    """
    # Set up CSV
    prediction = rf1.predict(X_test)
    prediction_prob = rf1.predict_proba(X_test)

    # Combine customer_id, prediction, and prediction_prob into a pandas DataFrame
    result_df = pd.DataFrame({
    'customer_id': test.customer_id,
    'prediction': prediction,
    'prediction_prob_1': prediction_prob[:, 1],})

    # Export the DataFrame as a CSV file
    result_df.to_csv('result.csv', index=False)

    return

def plot_that_target(train):
    """
    Visualize the target variable.

    Parameters:
    -----------
    train: pandas DataFrame
        
    Returns:
    --------
    countplot of churn in the train data set
    """
    sns.countplot(data=train, x='churn_encoded')
    plt.title('Churn')
    plt.show

    return

def plot_mcharges_v_churn(train):
    """
    Visualize the monthly charges vs. churn.

    Parameters:
    -----------
    train: pandas DataFrame
        
    Returns:
    --------
    histogram of monthly charges vs. churn in the train data set
    """
    sns.histplot(data=train, x='monthly_charges', hue='churn_encoded')
    plt.title('Monthly charges vs. Churn')
    plt.show()

    return

def plot_tenure_v_churn(train):
    """
    Visualize tenure vs. churn.

    Parameters:
    -----------
    train: pandas DataFrame
        
    Returns:
    --------
    histogram of tenure vs. churn in the train data set
    """
    sns.histplot(data=train, x='tenure', hue='churn_encoded')
    plt.title('Tenure vs. Churn')
    plt.show()

    return

def plot_tech_v_churn(train):
    """
    Visualize tech support vs. churn.

    Parameters:
    -----------
    train: pandas DataFrame
        
    Returns:
    --------
    countplot of tenure vs. churn in the train data set
    """
    sns.countplot(data=train, x='tech_support_Yes', hue='churn_encoded')
    plt.title('Tech Support vs. Churn')
    plt.show()

    return

def plot_senior_v_churn(train):
    """
    Visualize whether or not someone is a senior citizen and if they churn.

    Parameters:
    -----------
    train: pandas DataFrame
        
    Returns:
    --------
    countplot of senior citizen vs. churn in the train data set
    """
    sns.countplot(data=train, x='senior_citizen_encoded', hue='churn_encoded')
    plt.title('Senior citizen vs. Churn')
    plt.show()

    return