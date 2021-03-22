import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
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

def load_base_data(trans = False, port = False, prof = False):
    '''
    loads uncleaned data files in.
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
    # df['all_offers'] = df['event_offer received'] + df['event_offer completed']
    df = member_length(df)

    return df

def predict_offers(df, trans_id):
    '''
    ARGS: df - Starbucks DataFrame

    RETURN: model_dict - Dictionary Models for each possible offer
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    '''
    pred_df = df[['offer_index', 'success', 'gender', 'age', 'income', 'member_length']].copy()
    offer_dict = {}
    model_dict = {}
    success_accuracy = []


    offer_list = pred_df['offer_index'].unique().tolist()
    for offer_num in offer_list:
        offer_dict[f"offer_{offer_num}"] = pred_df[pred_df['offer_index'] == offer_num].copy()
        offer_dict[f'offer_{trans_id}'] = df[['offer_index', 'success', 'gender', 'age', 'income', 'member_length']].copy()
        X = offer_dict[f'offer_{offer_num}'].iloc[:,2:].copy()

        X = pd.get_dummies(X)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(X.values)

        X = pd.DataFrame(x_scaled, columns = X.columns)
        y = offer_dict[f'offer_{offer_num}'].iloc[:,1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        success_accuracy.append([offer_num , model.score(X_test, y_test)])
        model_dict[f'offer_{offer_num}'] = model

    return model_dict, success_accuracy

def predict_amount(df, trans_id):
    '''
    ARGS: df - Starbucks DataFrame

    RETURN: model_dict - Dictionary Models for each possible offer
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    '''
    pred_df = df[['offer_index', 'amount', 'gender', 'age', 'income', 'member_length']].copy()
    offer_dict = {}
    model_dict = {}
    amount_r2 = []

    offer_list = pred_df['offer_index'].unique().tolist()
    # for n in range(1, (pred_df['offer_index'].nunique() + 1)):
    for offer_num in offer_list:
        offer_dict[f"offer_{offer_num}"] = pred_df[pred_df['offer_index'] == offer_num].copy()
        offer_dict[f'offer_{trans_id}'] = df[['offer_index', 'amount', 'gender', 'age', 'income', 'member_length']].copy()
        X = offer_dict[f'offer_{offer_num}'].iloc[:,2:].copy()

        X = pd.get_dummies(X)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(X.values)

        X = pd.DataFrame(x_scaled, columns = X.columns)
        y = offer_dict[f'offer_{offer_num}'].iloc[:,1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        model = Ridge()
        model.fit(X_train, y_train)
        amount_r2.append([offer_num , model.score(X_test, y_test)])
        model_dict[f'offer_{offer_num}'] = model

    return model_dict, amount_r2

def pre_pred_data(df):
    '''
    ARGS : df - DataFrame

    RETURNS: df - DataFrame
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Takes Starbucks DataFrame and prepares it for prediction using models by
    filtering out redundant columns and scaling data.
    '''
    # Get each unique user's features
    pre_pred_df_1 = df.groupby(['person_index', 'gender', 'age', 'income', \
                            'member_length']).count().reset_index().iloc[:,:5].copy()
    pre_pred_df = pd.get_dummies(pre_pred_df_1).copy()
    # Scale
    min_max_scaler = preprocessing.MinMaxScaler()
    user_scaled = min_max_scaler.fit_transform(pre_pred_df.iloc[:,1:].values)
    pre_pred_df = pd.DataFrame(user_scaled, columns = pre_pred_df.iloc[:,1:].columns)
    pre_pred_df['person_index'] = pre_pred_df_1['person_index'].values

    # re-order for no other reason than I find it slightly easier to read.
    pre_pred_df = pre_pred_df[['person_index','age', 'income', 'member_length', 'gender_F',\
                                'gender_M', 'gender_O']].copy()
    return pre_pred_df

def predict_all(full_df, offer_dict, amount_dict, person_idx, offer_idx_list):
    '''
    ARGS:
    df          - Specific DataFrame
    full_df     - Full DataFrame with all known persons
    offer_dict  - Dictionary of offer success models
    amount_dict - Dictionary of offer amount models
    person_idx  - Int index of person
    new_person  - If desired person prediction is not currently in database
    np_values   - Values of new person. Must be a list and contain:
                  age           - (int) eg. 57
                  income        - (float) eg. 45000.0
                  member_length - (int in days)' eg. 365
                  gender        - 'F' - Female
                                  'M' - Male
                                  'O' - Other
                  Order is very important.
    RETURNS:
    predict_df  - DataFrame of predicted values next to corresponding offer

    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Predicts success and transaction amount for every potential offer and returning
    a DataFrame in order of transaction amount.
    '''

    pred_cols = ['age', 'income', 'member_length', 'gender_F', 'gender_M', 'gender_O']
    user_value = full_df[full_df.person_index == person_idx].loc[:,pred_cols]
    offer_preds = []
    offer_amount = []

    for offer_num in offer_idx_list:

        offer_preds.extend(offer_dict[f'offer_{offer_num}'].predict(user_value))
        offer_amount.extend(amount_dict[f'offer_{offer_num}'].predict(user_value))

    d = {'offer_index' : offer_idx_list, 'offer_success' : offer_preds,
        'predicted_amount' : offer_amount}

    predict_df = pd.DataFrame(d)
    predict_df.predicted_amount = predict_df.predicted_amount * predict_df.offer_success
    predict_df.predicted_amount = predict_df.predicted_amount.round(2)

    return predict_df, user_value

def predictor(df, full_df, person_idx):
    '''
    ARGS:
    full_df     - Full DataFrame for user values
    df          - DataFrame to train models on
    person_idx  - index of person to be predicted
    offer_dict  - Dictionary of offer success models
    amount_dict - Dictionary of offer amount models

    RETURNS:
    predict_df  - DataFrame of predicted values next to corresponding offer

    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Predicts success and transaction amount for every potential offer and returning
    a DataFrame in order of transaction amount.
    '''


    user_df = pre_pred_data(df)
    full_user_df = pre_pred_data(full_df)

    non_purchase_data = full_df[full_df['success'] == 0].copy()
    purchase_data = full_df[full_df['success'] == 1].copy()
    random_pd = purchase_data.sample(non_purchase_data.shape[0],random_state = 42).copy()
    new_train_data = pd.concat([non_purchase_data, random_pd]).reset_index(drop=True)


    portfolio = df[['offer_reward', 'difficulty', 'duration',
                        'offer_type', 'offer_id', 'email', 'mobile', 'social',
                        'web', 'offer_index']].copy()

    offers = portfolio.groupby(['offer_index', 'offer_reward', 'difficulty',
                        'duration', 'offer_type', 'offer_id', 'email', 'mobile',
                        'social', 'web']).count().reset_index()

    offer_list = df['offer_index'].unique().tolist()
    user_data = full_df[full_df.person_index == person_idx]

    try:
        trans_id = offers[offers['offer_type'] == 'transaction'].iloc[0,0]

        success_model, success_accuracy = predict_offers(df, trans_id)
        amount_model, amount_r2 = predict_amount(df, trans_id)

        pred, user_value = predict_all(full_user_df, offer_dict = success_model, amount_dict = amount_model, person_idx = person_idx, offer_idx_list = offer_list)
        pred = pred.sort_values(by = 'predicted_amount', ascending = False).reset_index(drop = True)

        return success_accuracy, amount_r2, pred, offers, user_value, user_data, success_model, amount_model

    except:

        print("I'm sorry, but a prediction cannot be made for this person.")
        print("It looks like this person does not exist in the current database.")

        return None

# save_data()
#
# df = load_data(filename = './data/sb_hour6.pkl')
# full_df = load_data()
# predictor(df, full_df, 4)


'''
For next time.
Predict Best For each hour

For Now:
- Build Class.


new_person  - If desired person prediction is not currently in database
np_values   - Values of new person. Must be a list and contain:
              age           - (int) eg. 57
              income        - (float) eg. 45000.0
              member_length - (int in days)' eg. 365
              gender        - 'F' - Female
                              'M' - Male
                              'O' - Other
              Order is very important.
'''
# test = pd.read_pickle('amount_member_length_test.pickle')
# for n in range(1, 12):
#     test.T[n].plot()
#
#
# test = pd.read_pickle('success_gender_test.pickle')
# for n in range(1, 12):
#     test.T[n].plot()
#
# non_purchase_data = full_df[full_df['success'] == 0].copy()
# purchase_data = full_df[full_df['success'] == 1].copy()
#
# len(random_pd)
# len(non_purchase_data)
#
# random_pd = purchase_data.sample(non_purchase_data.shape[0],random_state = 42).copy()
# new_train_data = pd.concat([non_purchase_data, random_pd]).reset_index(drop=True)
#
# full_training = pd.DataFrame(columns = full_df.columns)
# for n in range(1,12):
#     non_purchase_data = full_df[(full_df['offer_index'] == n) & (full_df['success'] == 0)].copy()
#     purchase_data = full_df[(full_df['offer_index'] == n) & (full_df['success'] == 1)].copy()
#     if non_purchase_data.shape[0] > purchase_data.shape[0]:
#         random_npd = non_purchase_data.sample(purchase_data.shape[0],random_state = 42).copy()
#         new_train_data = pd.concat([purchase_data, random_npd]).reset_index(drop=True)
#
#     elif purchase_data.shape[0] > non_purchase_data.shape[0]:
#         random_pd = purchase_data.sample(non_purchase_data.shape[0],random_state = 42).copy()
#         new_train_data = pd.concat([non_purchase_data, random_pd]).reset_index(drop=True)
#
#     full_training = full_training.append(new_train_data)
#
# full_training.shape[0]
# full_df[full_df['offer_index'] == 3].shape[0]
#
# full_df.groupby(['offer_index','success']).count()
# full_training.groupby(['offer_index','success']).count()
# #set X variable and get dummy variables for each feature within
# X = new_train_data.iloc[:,3:].copy()
# X = pd.get_dummies(data=X, columns=['V1','V4', 'V5','V6','V7'], drop_first = True)
# y = new_train_data['purchase']
#
# test[['success']].value_counts()
