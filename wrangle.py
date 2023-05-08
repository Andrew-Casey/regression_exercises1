import pandas as pd
import numpy as np
import env
import os
import wrangle as w
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def check_file_exists(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe
    """
    if os.path.isfile(fn):
        print('csv file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else: 
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df 


def get_zillow():
    url = env.get_db_url('zillow')
    query = ''' select bedroomcnt, 
                       bathroomcnt, 
                       calculatedfinishedsquarefeet, 
                       taxvaluedollarcnt, 
                       yearbuilt, 
                       taxamount, 
                       fips
                from properties_2017
                    join propertylandusetype using (propertylandusetypeid)
                    where propertylandusedesc Like 'Single Family Residential'
             '''
    filename = 'zillow.csv'
    df = check_file_exists(filename, query, url)

    return df

def remove_outliers(df, exclude_column=None, threshold=3):
    """
    This function removes outliers from a pandas dataframe, with an option to exclude a single column.
    Args:
    df: pandas dataframe
    exclude_column: string, optional column name to exclude from outlier detection
    threshold: float, optional number of standard deviations from the mean to consider a value an outlier
    Returns:
    pandas dataframe with outliers removed
    """
    if exclude_column is not None:
        # Copy dataframe and drop excluded column
        df_clean = df.drop(columns=exclude_column)
    else:
        df_clean = df.copy()
    # Calculate z-score for each value
    z_scores = np.abs((df_clean - df_clean.mean()) / df_clean.std())
    # Remove rows with any value above threshold
    df_clean = df.loc[(z_scores < threshold).all(axis=1)]
    return df_clean

def wrangle_zillow():
    
    #load zillow database
    df = w.get_zillow()
    
    #rename columns
    df = df.rename(columns={'bedroomcnt': 'bedrooms'
                        , 'bathroomcnt': 'bathrooms'
                        , 'calculatedfinishedsquarefeet': 'sqft'
                        , 'taxvaluedollarcnt':'taxvalue'
                        ,'yearbuilt': 'built'
                        , 'taxamount':'tax'})
    # change fips codes to county name
    df['fips'] = df['fips'].replace([6037.0, 6059.0, 6111.0],['LA','Orange','Ventura']).astype(str)

    #move my target variable taxvalue to the 1st column in the dataframe
    column_to_move = df.pop("taxvalue")
    df.insert(0, "taxvalue", column_to_move)
    df
     # Create dummy variables for the species column
    dummy_df = pd.get_dummies(df['fips'], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    #drop all nulls
    df = df.dropna()
    
    #handle outliers
    df = df[df.taxvalue <= 2000000]  
    df = df[df.sqft <= 5000]
    df = df[df.tax <= 30000]

    return df

from sklearn.model_selection import train_test_split

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


def wrangle_zillow2():
    
    #load zillow database
    df = w.get_zillow()
    
    #rename columns
    df = df.rename(columns={'bedroomcnt': 'rooms'
                        , 'bathroomcnt': 'bath'
                        , 'calculatedfinishedsquarefeet': 'sqft'
                        , 'taxvaluedollarcnt':'taxvalue'
                        ,'yearbuilt': 'built'
                        , 'taxamount':'tax'})
    #move my target variable taxvalue to the 1st column in the dataframe
    column_to_move = df.pop("taxvalue")
    df.insert(0, "taxvalue", column_to_move)
    df
    #drop all nulls
    df = df.dropna()
    
    #handle outliers
    df = w.remove_outliers(df, exclude_column='fips', threshold=4)

    return df

def wrangle_zillow3():
    
    #load zillow database
    df = w.get_zillow()
    
    #rename columns
    df = df.rename(columns={'bedroomcnt': 'rooms'
                        , 'bathroomcnt': 'bath'
                        , 'calculatedfinishedsquarefeet': 'sqft'
                        , 'taxvaluedollarcnt':'taxvalue'
                        ,'yearbuilt': 'built'
                        , 'taxamount':'tax'})
    #move my target variable taxvalue to the 1st column in the dataframe
    column_to_move = df.pop("taxvalue")
    df.insert(0, "taxvalue", column_to_move)
    df
    #drop all nulls
    df = df.dropna()
    
    #handle outliers
    df = w.remove_outliers(df, exclude_column='fips', threshold=3)

    return df

def scaled_df(train, validate, test):

    X_train = train[['bedrooms','bathrooms','sqft','built','tax','Orange','Ventura']]
    X_validate = validate[['bedrooms','bathrooms','sqft','built','tax','Orange','Ventura']]
    X_test = test[['bedrooms','bathrooms','sqft','built','tax','Orange','Ventura']]

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
    X_train_scaled = df_X_train_scaled.rename(columns={0: 'bedrooms', 1: 'bathrooms', 2: 'sqft', 3: 'built', 4: 'tax', 5: 'Orange', 6:'Ventura'})

    # Convert the array to a DataFrame
    df_X_validate_scaled = pd.DataFrame(X_validate_scaled)
    X_validate_scaled = df_X_validate_scaled.rename(columns={0: 'bedrooms', 1: 'bathrooms', 2: 'sqft', 3: 'built', 4: 'tax', 5: 'Orange', 6:'Ventura'})
    
    # Convert the array to a DataFrame
    df_X_test_scaled = pd.DataFrame(X_test_scaled)
    X_test_scaled = df_X_test_scaled.rename(columns={0: 'bedrooms', 1: 'bathrooms', 2: 'sqft', 3: 'built', 4: 'tax', 5: 'Orange', 6:'Ventura'})
    
    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test