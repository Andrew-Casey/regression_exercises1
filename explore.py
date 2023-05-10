import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt

import env
import os
import wrangle as w

def plot_variable_pairs(train):
    sns.set(style="ticks")
    sns.pairplot(train, kind="reg", corner = True, hue='fips', plot_kws={'line_kws': {'color': 'red'}})
    plt.show()



def plot_categorical_and_continuous_vars(dataframe, categorical_var, continuous_var):
    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=dataframe[categorical_var], y=dataframe[continuous_var])
    plt.xlabel(categorical_var)
    plt.ylabel(continuous_var)
    plt.title(f"Box Plot of {continuous_var} vs {categorical_var}")
    plt.show()

    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.stripplot(x=dataframe[categorical_var], y=dataframe[continuous_var])
    plt.xlabel(categorical_var)
    plt.ylabel(continuous_var)
    plt.title(f"Strip Plot of {continuous_var} vs {categorical_var}")
    plt.show()

    # Swarm plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=dataframe[categorical_var], y=dataframe[continuous_var])
    plt.xlabel(categorical_var)
    plt.ylabel(continuous_var)
    plt.title(f"Bar Plot of {continuous_var} vs {categorical_var}")
    plt.show()

