import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pickle

from clean_data import *
from process_data import *

def member_length(df):

    '''
    ARGS:   df - DataFrame

    RETURN: df - DataFrame
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

def load_base_data(trans = False, port = False, prof = False):
    '''
    ARGS: trans - (default: False) if True, returns transaction DataFrame
          port  - (default: False) if True, returns portfolio DataFrame
          prof  - (default: False) if True, returns profile DataFrame

    RETURNS: transcript_df, portfolio_df, profile_df as default or specific
             mix depending on inputs.
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Loads core/unprocessed data files for clarification purposes or checking I
    didn't go way off the mark.
    '''
    portfolio_df_dirty = pd.read_json('data/portfolio.json', lines = True)
    profile_df_dirty = pd.read_json('data/profile.json', lines = True)
    transcript_df_dirty = pd.read_json('data/transcript.json', lines = True)

    # Clean data
    profile_df = clean_profile_df(profile_df_dirty)
    transcript_df = clean_transcript_df(transcript_df_dirty)
    portfolio_df = clean_portfolio_df(portfolio_df_dirty)

    transcript_df, portfolio_df, profile_df  = id_simpify(transcript_df, portfolio_df, profile_df)

    if trans == True:
        return transcript_df
    if port == True:
        return portfolio_df
    if prof == True:
        return profile_df
    else:
        return transcript_df, portfolio_df, profile_df

def load_data(filename = './data/clean_data.pkl'):
    '''
    ARGS: None

    RETURNS: df - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Loads in data into DataFrame and applies the final touches of preperation
    from processing.

    '''
    df = pd.read_pickle(filename)
    df = member_length(df)

    return df

def predict_success(df, trans_id):
    '''
    ARGS:   df         - Starbucks DataFrame
            trans_id   - Offer Index of Transaction - it occassionally changes
                         if input file also changes so this makes sure to keep
                         track of it.

    RETURN: model_dict       - Dictionary of models for each possible offer
            success_accuracy - accuracy score of each success model in model_dict
            amount_mse       - F1 score of each success model in model_dict
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Uses Logistic Regression to predict success of an offer based off of
    demographic information alone.
    '''

    pred_df = df[['offer_index', 'success', 'gender', 'age', 'income', 'member_length']].copy()
    offer_dict = {}
    model_dict = {}
    success_accuracy = []
    success_f1 = []

    offer_list = pred_df['offer_index'].unique().tolist()
    for offer_num in offer_list:
        offer_dict[f"offer_{offer_num}"] = pred_df[pred_df['offer_index'] == offer_num].copy()
        offer_dict[f'offer_{trans_id}'] = df[['offer_index', 'success', 'gender', 'age', 'income', 'member_length']].copy()

        X = offer_dict[f'offer_{offer_num}'].iloc[:,2:].copy()
        X = pd.get_dummies(X)

        y = offer_dict[f'offer_{offer_num}'].iloc[:,1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        success_accuracy.append([offer_num, model.score(X_test, y_test)])
        y_pred = model.predict(X_test)
        success_f1.append([offer_num, f1_score(y_test, y_pred)])

        model_dict[f'offer_{offer_num}'] = model

    return model_dict, success_accuracy, success_f1

def predict_amount(df, trans_id):
    '''
    ARGS:   df         - Starbucks DataFrame
            trans_id   - Offer Index of Transaction - it occassionally changes
                         if input file also changes so this makes sure to keep
                         track of it.

    RETURN: model_dict - Dictionary Models for each possible offer
            amount_r2  - r2 score of each amount model in model_dict
            amount_mse - mean_squared_error of each amount model in model_dict
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Uses Ridge Regression to predict potential transaction amount of an offer
    based off of demographic information alone.
    '''
    df = df[df['success'] == 1]
    pred_df = df[['offer_index', 'amount', 'gender', 'age', 'income', 'member_length']].copy()

    offer_dict = {}
    model_dict = {}
    amount_r2 = []
    amount_mse = []

    offer_list = pred_df['offer_index'].unique().tolist()

    for offer_num in offer_list:
        offer_dict[f"offer_{offer_num}"] = pred_df[pred_df['offer_index'] == offer_num].copy()
        offer_dict[f'offer_{trans_id}'] = df[['offer_index', 'amount', 'gender', 'age', 'income', 'member_length']].copy()
        X = offer_dict[f'offer_{offer_num}'].iloc[:,2:].copy()

        X = pd.get_dummies(X)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X.values)

        X = pd.DataFrame(x_scaled, columns = X.columns)
        y = offer_dict[f'offer_{offer_num}'].iloc[:,1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        amount_r2.append([offer_num , model.score(X_test, y_test)])
        amount_mse.append([offer_num, mean_squared_error(y_test, model.predict(X_test))])
        model_dict[f'offer_{offer_num}'] = model

    return model_dict, amount_r2, amount_mse

def pre_scale(df):
    '''
    ARGS : df - DataFrame

    RETURNS: pre_scale_df - DataFrame
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Takes Starbucks DataFrame and prepares it for prediction using models by
    filtering out redundant columns and scaling data.
    '''
    # Get each unique user's features
    pre_pred_users = df.groupby(['person_index', 'gender', 'age', 'income','member_length']).count().reset_index().iloc[:,:5].copy()
    pre_scale_df = pd.get_dummies(pre_pred_users).copy()

    # Scale Features
    scaler = StandardScaler()
    user_scaled = scaler.fit_transform(pre_scale_df.iloc[:,1:].values)
    pre_scale_df = pd.DataFrame(user_scaled, columns = pre_scale_df.iloc[:,1:].columns)
    pre_scale_df['person_index'] = pre_pred_users['person_index'].values

    # re-order for no other reason than I find it slightly easier to read.
    pre_scale_df = pre_scale_df[['person_index','age', 'income', 'member_length', 'gender_F',\
                                'gender_M', 'gender_O']].copy()
    return pre_scale_df

def build_models(df):
    '''
    ARGS:
            df                - DataFrame
    RETURNS:
            success_accuracy  - accuracy score of success_model
            success_f1        - f1 score of success_model
            amount_r2         - r2 score of amount_model
            amount_mse        - mean_squared_error of amount model
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Creates two model dictionaries containing predictions for all offers.

    '''
    offers = df.groupby(['offer_index', 'offer_reward', 'difficulty',
                        'duration', 'offer_type', 'offer_id', 'email', 'mobile',
                        'social', 'web']).count().reset_index().iloc[:,:10].copy()

    trans_id = offers[offers['offer_type'] == 'transaction'].iloc[0,0]

    success_model, success_accuracy, success_f1 = predict_success(df, trans_id)
    amount_model, amount_r2, amount_mse = predict_amount(df, trans_id)

    return success_model, success_accuracy, success_f1, amount_model, amount_r2, amount_mse, offers

def predictor(df, person_idx, success_model_dict, amount_model_dict):
    '''
    ARGS:
            df                - DataFrame
            person_idx        - index of person to be predicted
            offer_dict        - Dictionary of offer success models
            amount_dict       - Dictionary of offer amount models

    RETURNS:
            predict           - DataFrame of predicted success and amount_s
                                next to corresponding offer

            user_value        - Standardised user demographic info for prediction
            user_data         - All rows from full dataframe corresponding to person_idx
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Predicts success and transaction amount for every potential offer and returning
    a DataFrame in order of transaction amount.
    '''

    unscaled_df  = df.groupby(['person_index', 'gender', 'age', 'income','member_length']).count().reset_index().iloc[:,:5].copy()
    unscaled_df = pd.get_dummies(unscaled_df)

    scaled_df = pre_scale(df)

    user_data = unscaled_df[unscaled_df.person_index == person_idx]

    # build data
    pred_cols = ['age', 'income', 'member_length', 'gender_F', 'gender_M', 'gender_O']
    suc_user_value = unscaled_df[unscaled_df.person_index == person_idx].loc[:, pred_cols]
    am_user_value  = scaled_df[scaled_df.person_index == person_idx].loc[:, pred_cols]

    # use models
    offer_success = []
    offer_amount = []
    offer_list = list(range(1,12,1))
    for offer_num in offer_list:
        offer_success.extend(success_model_dict[f'offer_{offer_num}'].predict(suc_user_value))
        offer_amount.extend(amount_model_dict[f'offer_{offer_num}'].predict(am_user_value))

    # store results
    d = {'offer_index' : offer_list, 'predicted_success' : offer_success,
        'predicted_amount' : offer_amount}

    predict_df = pd.DataFrame(d)
    predict_df.predicted_amount = predict_df.predicted_amount.round(2)
    predict_df = predict_df.sort_values(by = 'offer_index', ascending = True).reset_index(drop = True)

    return  predict_df, suc_user_value, user_data


def new_person_check(person_index, age, income, membership_length, gender):
    '''
    ARGS :
    person_index - (int) Index for New Person, must be unique and not current
                   person database
    age          - (int) Age for New Person, must be older than 0 and younger that 120
    income       - (int/float) Income for New Person, must be between 0 and 140000.0
    membership_length - (int) Membership length in days
    gender       - (str) Gender - 'F' = Female
                                - 'M' = Male
                                - 'O' = Other
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Performs checks on values used for creating a new user for prediction
    '''
    assert isinstance(age, int)
    assert (age > 0) & (age < 120)

    assert isinstance(income, int) | isinstance(income, float)
    assert (income > 0) & (income < 140000)

    assert isinstance(membership_length, int)
    assert (membership_length >= 0)

    assert isinstance(gender, str)
    assert gender in ['F','M','O','f','m','o']

    if gender in ['F','f']:
        gender = 'F'
    elif gender in ['M','m']:
        gender = 'M'
    elif gender in ['O','o']:
        gender = 'O'
    values_checked = [person_index, age, income, membership_length, gender]
    return values_checked
