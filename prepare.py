import pandas as pd
import numpy as np


import wrangle as w
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt


def split_data(df):
    '''
    take in a DataFrame and target variable. return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123)
    return train, validate, test


def scaled_df(train, validate, test):

    X_train = train[['bedrooms','bathrooms','sqft','built','Orange','Ventura']]
    X_validate = validate[['bedrooms','bathrooms','sqft','built','Orange','Ventura']]
    X_test = test[['bedrooms','bathrooms','sqft','built','Orange','Ventura']]

    y_train = train.taxvalue
    y_validate = validate.taxvalue
    y_test = test.taxvalue

    #making our scaler
    scaler = MinMaxScaler()
    #fitting our scaler 
    # AND!!!!
    #using the scaler on train
    X_train_scaled = scaler.fit_transform(X_train)
    #using our scaler on validate
    X_validate_scaled = scaler.transform(X_validate)
    #using our scaler on test
    X_test_scaled = scaler.transform(X_test)

    # Convert the array to a DataFrame
    df_X_train_scaled = pd.DataFrame(X_train_scaled)
    X_train_scaled = df_X_train_scaled.rename(columns={0: 'bedrooms', 1: 'bathrooms', 2: 'sqft', 3: 'built', 4: 'Orange', 5:'Ventura'})

    # Convert the array to a DataFrame
    df_X_validate_scaled = pd.DataFrame(X_validate_scaled)
    X_validate_scaled = df_X_validate_scaled.rename(columns={0: 'bedrooms', 1: 'bathrooms', 2: 'sqft', 3: 'built', 4: 'Orange', 5:'Ventura'})
    
    # Convert the array to a DataFrame
    df_X_test_scaled = pd.DataFrame(X_test_scaled)
    X_test_scaled = df_X_test_scaled.rename(columns={0: 'bedrooms', 1: 'bathrooms', 2: 'sqft', 3: 'built', 4: 'Orange', 5:'Ventura'})
    
    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test

def scaled_df_tips(train, validate, test):

    X_train = train[['total_bill', 'sex', 'smoker', 'time', 'size','price_per_person', 'Fri', 'Sat', 'Sun','Thur']]
    X_validate = validate[['total_bill', 'sex', 'smoker', 'time', 'size','price_per_person', 'Fri', 'Sat', 'Sun','Thur']]
    X_test = test[['total_bill', 'sex', 'smoker', 'time', 'size','price_per_person', 'Fri', 'Sat', 'Sun','Thur']]

    y_train = train.tip
    y_validate = validate.tip
    y_test = test.tip

    #making our scaler
    scaler = MinMaxScaler()
    #fitting our scaler 
    # AND!!!!
    #using the scaler on train
    X_train_scaled = scaler.fit_transform(X_train)
    #using our scaler on validate
    X_validate_scaled = scaler.transform(X_validate)
    #using our scaler on test
    X_test_scaled = scaler.transform(X_test)

    # Convert the array to a DataFrame
    df_X_train_scaled = pd.DataFrame(X_train_scaled)
    X_train_scaled = df_X_train_scaled.rename(columns={0: 'total_bill', 1: 'sex', 2: 'smoker', 3: 'time', 4:'size', 5:'price_per_person', 6:'Fri', 7:'Sat', 8:'Sun', 9:'Thur'})

    # Convert the array to a DataFrame
    df_X_validate_scaled = pd.DataFrame(X_validate_scaled)
    X_validate_scaled = df_X_validate_scaled.rename(columns={0: 'total_bill', 1: 'sex', 2: 'smoker', 3: 'time', 4:'size', 5:'price_per_person', 6:'Fri', 7:'Sat', 8:'Sun', 9:'Thur'})
    
    # Convert the array to a DataFrame
    df_X_test_scaled = pd.DataFrame(X_test_scaled)
    X_test_scaled = df_X_test_scaled.rename(columns={0: 'total_bill', 1: 'sex', 2: 'smoker', 3: 'time', 4:'size', 5:'price_per_person', 6:'Fri', 7:'Sat', 8:'Sun', 9:'Thur'})
    
    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test

	

def scaled_df_swiss(train, validate, test):

    X_train = train[['Agriculture', 'Examination', 'Education', 'Catholic', 'Infant.Mortality']]
    X_validate = validate[['Agriculture', 'Examination', 'Education', 'Catholic', 'Infant.Mortality']]
    X_test = test[['Agriculture', 'Examination', 'Education', 'Catholic', 'Infant.Mortality']]

    y_train = train.Fertility
    y_validate = validate.Fertility
    y_test = test.Fertility

    #making our scaler
    scaler = MinMaxScaler()
    #fitting our scaler 
    # AND!!!!
    #using the scaler on train
    X_train_scaled = scaler.fit_transform(X_train)
    #using our scaler on validate
    X_validate_scaled = scaler.transform(X_validate)
    #using our scaler on test
    X_test_scaled = scaler.transform(X_test)

    # Convert the array to a DataFrame
    df_X_train_scaled = pd.DataFrame(X_train_scaled)
    X_train_scaled = df_X_train_scaled.rename(columns={0: 'Agriculture', 1: 'Examination', 2: 'Education', 3: 'Catholic', 4:'Infant.Mortality'})

    # Convert the array to a DataFrame
    df_X_validate_scaled = pd.DataFrame(X_validate_scaled)
    X_validate_scaled = df_X_validate_scaled.rename(columns={0: 'Agriculture', 1: 'Examination', 2: 'Education', 3: 'Catholic', 4:'Infant.Mortality'})
    
    # Convert the array to a DataFrame
    df_X_test_scaled = pd.DataFrame(X_test_scaled)
    X_test_scaled = df_X_test_scaled.rename(columns={0: 'Agriculture', 1: 'Examination', 2: 'Education', 3: 'Catholic', 4:'Infant.Mortality'})
    
    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test