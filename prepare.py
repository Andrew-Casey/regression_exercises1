import acquire as acq
import pandas as pd
from sklearn.model_selection import train_test_split

def prep_iris():

    #Load the iris data
    iris = acq.get_iris_data()

    # Drop the unnecessary columns
    iris = iris.drop(columns=['species_id', 'measurement_id'])

    # Rename the species_name column
    iris = iris.rename(columns={'species_name': 'species'})

    # Create dummy variables for the species column
    dummy_df = pd.get_dummies(iris['species'], drop_first=True)
    iris = pd.concat([iris, dummy_df], axis=1)

    return iris


def prep_titanic():
   
    # Load the titanic data
    titanic = acq.get_titanic_data()
    
    # Drop unnecessary columns
    titanic = titanic.drop(columns=['deck', 'embark_town', 'class'])
    #change pclass to an object
    #titanic['pclass'] = titanic['pclass'].astype(object)
    # Create dummy variables
    dummy_cols = ['sex', 'embarked']
    dummy_df = pd.get_dummies(titanic[dummy_cols], drop_first=True)
    titanic = pd.concat([titanic, dummy_df], axis=1)

    return titanic


def prep_telco():
    # Load the telco data
    telco = acq.get_telco_churn()
    # Drop unnecessary columns
    telco.drop(columns=['internet_service_type_id','contract_type_id','payment_type_id'])

    #Encode 
    telco['gender_encoded'] = telco['gender'].replace({'Female': 0, 'Male': 1})
    telco['senior_citizen_encoded'] = (telco['senior_citizen'] == 1).astype(int)
    telco['partner_encoded'] = telco['partner'].replace({'Yes': 1, 'No': 0})
    telco['dependents_encoded'] = telco['dependents'].replace({'Yes': 1, 'No': 0})
    telco['phone_service_encoded'] = telco['phone_service'].replace({'Yes': 1, 'No': 0})
    telco['paperless_billing_encoded'] = telco['paperless_billing'].replace({'Yes': 1, 'No': 0})
    telco['churn_encoded'] = telco['churn'].replace({'Yes': 1, 'No': 0})
    telco['total_charges'] = telco['total_charges'].replace(' ', 0).astype(float)

    #dummy variables
    dummy_cols = ['contract_type','internet_service_type','payment_type', 'multiple_lines', 'online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies']
    dummy_df = pd.get_dummies(telco[dummy_cols], drop_first=True)
    telco = pd.concat([telco, dummy_df], axis=1)

    return telco
#split function
def split_data(df, target):
    '''
    take in a DataFrame and target variable. return train, validate, and test DataFrames; stratify on target variable.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train_validate[target])
    return train, validate, test


