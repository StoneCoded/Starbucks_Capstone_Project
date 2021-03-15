import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, Perceptron, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from scipy.stats.stats import pearsonr
from sklearn import preprocessing
import pickle

from clean_data import *
from process_data import *

def member_length(df):
    ordinal_today = pd.to_datetime(datetime.today().strftime('%m/%d/%Y')).toordinal()
    df['member_length'] = ordinal_today - df['ordinal_ms']
    mem_diff = (df['member_length'].min() - df['days_elapsed'].max().round()).astype(int)
    df['member_length'] = df['member_length'] - mem_diff
    df = df.drop(['ordinal_ms'], axis = 1)
    return df

starbucks_df = pd.read_pickle('./dropped_clean_data.pkl')

starbucks_df['event_offer completed'] = np.where((starbucks_df['event_offer received'] == 0) & (starbucks_df['event_transaction'] == 0), 1, 0)
starbucks_df['all_offers'] = starbucks_df['event_offer received'] + starbucks_df['event_offer completed']


def predict_offers(df):
    '''
    ARGS: df - Starbucks DataFrame

    RETURN: model_dict - Dictionary Models for each possible offer
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    '''
    df = starbucks_df.copy()
    user_df = df[df['offer_type'] != 'transaction'].copy()
    user_df = user_df[['offer_index', 'success', 'gender', 'age', 'income', 'member_length']].copy()

    offer_dict = {}
    model_dict = {}
    success_accuracy = []
    # offer_index 1 is transactions
    for n in range(1, (user_df['offer_index'].nunique() + 1)):
        offer_dict[f"offer_{n}"] = user_df[user_df['offer_index'] == n].copy()
        offer_dict['offer_6'] = user_df.copy()
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


        return model_dict

model_dict['offer_1'].predict(X_test)
coefs = model_dict['offer_1'].coef_

'''
For next time.
Build person class that can be predicted upon.
We want to show the chance of success by each offer type
    - extending this if we can predict amount spent
    - Also maybe best chance of succes by time sent out

For Now:
- Predict Amount for Each


'''
