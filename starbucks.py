import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import pickle

from clean_data import *
from process_data import *

def member_length(df):
    '''
    ARGS: DataFrame

    RETURN: DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Converts the ordinal membership start date into a duration of membership in
    days.
    '''
    ordinal_today = pd.to_datetime(datetime.today().strftime('%m/%d/%Y')).toordinal()
    df['member_length'] = ordinal_today - df['ordinal_ms']
    mem_diff = (df['member_length'].min() - df['days_elapsed'].max().round()).astype(int)
    df['member_length'] = df['member_length'] - mem_diff
    df = df.drop(['ordinal_ms'], axis = 1)
    return df

def load_data():
    '''
    ARGS: None

    RETURNS: df - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Loads in data into DataFrame and applies the final touches of preperation
    from processing.

    '''
    df = pd.read_pickle('./dropped_clean_data.pkl')

    df['event_offer completed'] = np.where((df['event_offer received'] == 0) & (df['event_transaction'] == 0), 1, 0)
    df['all_offers'] = df['event_offer received'] + df['event_offer completed']

    df = member_length(df)

    return df

def predict_offers(df):
    '''
    ARGS: df - Starbucks DataFrame

    RETURN: model_dict - Dictionary Models for each possible offer
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    '''
    user_df = df[df['offer_type'] != 'transaction'].copy()
    user_df = user_df[['offer_index', 'success', 'gender', 'age', 'income', 'member_length']].copy()

    offer_dict = {}
    model_dict = {}
    success_accuracy = []
    # offer_index 6 is transactions
    for n in range(1, (user_df['offer_index'].nunique() + 2)):
        offer_dict[f"offer_{n}"] = user_df[user_df['offer_index'] == n].copy()
        offer_dict['offer_6'] = df[['offer_index', 'success', 'gender', 'age', 'income', 'member_length']].copy()
        X = offer_dict[f'offer_{n}'].iloc[:,2:].copy()

        X = pd.get_dummies(X)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(X.values)

        X = pd.DataFrame(x_scaled, columns = X.columns)
        y = offer_dict[f'offer_{n}'].iloc[:,1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        success_accuracy.append([n , model.score(X_test, y_test)])
        model_dict[f'offer_{n}'] = model

    return model_dict, success_accuracy

def predict_amount(df):
    '''
    ARGS: df - Starbucks DataFrame

    RETURN: model_dict - Dictionary Models for each possible offer
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    '''
    df = starbucks_df.copy()
    user_df = df[df['offer_type'] != 'transaction'].copy()
    user_df = user_df[['offer_index', 'amount', 'gender', 'age', 'income', 'member_length']].copy()

    offer_dict = {}
    model_dict = {}
    success_r2 = []
    # offer_index 6 is transactions
    for n in range(1, (user_df['offer_index'].nunique() + 2)):
        offer_dict[f"offer_{n}"] = user_df[user_df['offer_index'] == n].copy()
        offer_dict['offer_6'] = df[['offer_index', 'amount', 'gender', 'age', 'income', 'member_length']].copy()
        X = offer_dict[f'offer_{n}'].iloc[:,2:].copy()

        X = pd.get_dummies(X)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(X.values)

        X = pd.DataFrame(x_scaled, columns = X.columns)
        y = offer_dict[f'offer_{n}'].iloc[:,1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        success_r2.append([n , model.score(X_test, y_test)])
        model_dict[f'offer_{n}'] = model

    return model_dict, success_r2

starbucks_df = load_data()
offer_df = starbucks_df.groupby(['offer_index','offer_id','offer_type'])\
                                    .count().reset_index().iloc[:,:3].copy()

def pre_pred_data(df):
    '''
    ARGS : df - DataFrame

    RETURNS: df - DataFrame
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Takes Starbucks DataFrame and prepares it for prediction using models by
    filtering out redundant columns and scaling data.
    '''
    # Get each unique user's features
    user_df_1 = df.groupby(['person_index', 'gender', 'age', 'income', \
                            'member_length']).count().reset_index().iloc[:,:5].copy()
    user_df = pd.get_dummies(user_df_1).copy()
    # Scale
    min_max_scaler = preprocessing.MinMaxScaler()
    user_scaled = min_max_scaler.fit_transform(user_df.iloc[:,1:].values)
    user_df = pd.DataFrame(user_scaled, columns = user_df.iloc[:,1:].columns)
    user_df['person_index'] = user_df_1['person_index'].values

    # re-order for no other reason than I find it slightly easier to read.
    user_df = user_df[['person_index','age', 'income', 'member_length', 'gender_F',\
                                'gender_M', 'gender_O']].copy()
    return user_df


pred_cols = ['age', 'income', 'member_length', 'gender_F', 'gender_M', 'gender_O']
offer_dict, accuracy = predict_offers(starbucks_df)
amount_dict, r2 = predict_amount(starbucks_df)
user_df = pre_pred_data(starbucks_df)



def predict(user_index):

    user_value = user_df[user_df.person_index == user_index].loc[:,pred_cols]

    offer_num = []
    offer_preds = []
    offer_amount = []

    for n in range(1, 12):
        offer_num.append(n)
        offer_preds.extend(offer_dict[f'offer_{n}'].predict(user_value))
        offer_amount.extend(amount_dict[f'offer_{n}'].predict(user_value))
        d = {'offer_index' : offer_num, 'offer_success' : offer_preds,
            'predicted_amount' : offer_amount}
        predict_df = pd.DataFrame(d)
        
    return predict_df



user_df.columns


'''
For next time.
Build person class that can be predicted upon.
We want to show the chance of success by each offer type
    - Also maybe best chance of succes by time sent out

For Now:
- Build Class.


'''





amt_models, r2_scores = predict_amount(starbucks_df)

for n in range(1, 12):
    user_df[f'offer_{n}_amnt'] = amt_models[f'offer_{n}'].predict(user_df.iloc[:,1:7])

user_df.describe()

for n in range(1, 12):
    # user_df[f'offer_{n}'] = model_dict[f'offer_{n}'].predict(user_df.iloc[:,1:7])
    pred_cols = ['age', 'income', 'member_length', 'gender_F', 'gender_M', 'gender_O']
    user_idx = 1
    print((amt_models[f'offer_{n}'].predict(user_df.loc[1:3,pred_cols])))
