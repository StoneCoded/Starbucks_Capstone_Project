import pandas as pd
from datetime import datetime
from process_data import *
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

import pickle

'''
––––––––––––––––––––––Data File Descriptions––––––––––––––––––––––––
__portfolio.json__
id (string) - offer id
offer_type (string) - type of offer ie BOGO, discount, informational
difficulty (int) - minimum required spend to complete an offer
reward (int) - reward given for completing an offer
duration (int) - time for offer to be open, in days
channels (list of strings)

__profile.json__
age (int) - age of the customer
became_member_on (int) - date when customer created an app account
gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
id (str) - customer id
income (float) - customer's income

__transcript.json__
event (str) - record description (ie transaction, offer received, offer viewed, etc.)
person (str) - customer id
time (int) - time in hours since start of test. The data begins at time t=0
value - (dict of strings) - either an offer id or transaction amount depending on the record
–––––––––––––––––––––––––––File Overview–––––––––––––––––––––––––––––
Each fuction has its own doc string that describes the purpose of each.

Initally I wanted to save the model builds and use them further on but they are
HUGE files. So I will refactor this at a later date to disable the saving ability
but for now I will just export the cleaned data.
'''

def clean_transcript_df(df):
    take_available = lambda s1, s2: s1 if s1 == True else s2
    # Get values out of dict
    df['amount'] = [x.get('amount') for x in df.iloc[:, 2]]
    df['reward'] = [x.get('reward') for x in df.iloc[:, 2]]
    df['offer id'] = [x.get('offer id') for x in df.iloc[:, 2]]
    df['offer_id'] = [x.get('offer_id') for x in df.iloc[:, 2]]
    df['offer_id'].fillna(df['offer id'], inplace = True)
    df = df.drop(['value','offer id'], axis = 1)
    df['time'] = df['time'].div(24)
    # Duplicates hint user has received multiple of the same offer
    df = df.drop_duplicates()
    df.columns = ['person_id', 'event', 'days_elapsed', 'amount', 'reward', 'offer_id']
    return df
def clean_portfolio_df(df):
    # take channels out of lists and makes dummies of them
    portfolio_channels = pd.get_dummies(df.channels.apply(pd.Series).stack()).sum(level=0)
    df = df.join(portfolio_channels)
    df = df.drop('channels', axis = 1)
    df.columns = ['offer_reward', 'difficulty', 'duration', 'offer_type', \
                                    'offer_id', 'email', 'mobile', 'social', 'web']
    return df
def clean_profile_df(df):
    #where gender and income == NaN is also where age is 118, so dropping all
    #eranious values
    # df = df.gender.reset_index(drop = True)
    df.columns = ['gender', 'age', 'person_id', 'membership_start','income']

    df['membership_start'] = [datetime.strptime(str(x), '%Y%m%d').\
                        strftime('%m/%d/%Y') for x in df.membership_start]

    df['membership_start'] = pd.to_datetime(df.membership_start)

    return df
def age_brackets(df):
    '''
    ARGS:    df - DataFrame

    RETURNS :df - DataFrame
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Fits 'age' columns into 10 year bins to form a general age category
    '''
    bins = [18,25,35,45,55,65,75,85,95,105]
    labels = ['18-24','25-34','35-44','45-54','55-64','65-74','75-84','85-94','95-104']
    df['age_bracket'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    return df
def income_brackets(df):
    '''
    ARGS:    df - DataFrame

    RETURNS: df - DataFrame
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Fits 'income' rows into 3 year bins to form a general income category
    '''
    bins = [10000,50000,90000,130000]
    labels = ['low','medium','high']
    df['income_bracket'] = pd.cut(df['income'], bins=bins, labels=labels, right=False)

    return df


def id_simpify(transcript_df, portfolio_df, profile_df):
    '''
    ARGS:
    transcript_df - Transcript Dataframe
    portfolio_df  - Portfolio Dataframe
    profile_df    - Profile Dataframe

    RETURNS:
    df            - Dataframe
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Assigns each long id string with a simple unique integer id.

    For offer_ids, an int id is also assigned to transactions.

    Example:

    person_id                           person_index
    78afa995795e4d85b5d9ceeca43f5fef :  1713
    –––––––––––––––––––––––––––––––––––––––––––––––––
    offer_id                            offer_index
    9b98b8c7a33c4b65b9aebfe6a799e6d9 :  9

    '''
    offer_ids = portfolio_df.offer_id.unique().tolist()
    offer_ids_2 = transcript_df.offer_id.unique().tolist()

    person_ids = profile_df.person_id.unique().tolist()
    person_ids_2 = transcript_df.person_id.unique().tolist()

    offer_id_tot = list(set(offer_ids + offer_ids_2))
    person_id_tot = list(set(person_ids + person_ids_2))

    offer_id_sub = np.array(range(1,len(offer_id_tot)+1))
    offer_id_data = {'offer_id': offer_id_tot, 'offer_index': offer_id_sub}
    offer_encode_df = pd.DataFrame(offer_id_data)

    person_id_sub = np.array(range(1,len(person_id_tot)+1))
    person_id_data = {'person_id': person_id_tot, 'person_index': person_id_sub}
    person_encode_df = pd.DataFrame(person_id_data)

    profile_df = profile_df.merge(person_encode_df, on = 'person_id', how = 'left')
    portfolio_df = portfolio_df.merge(offer_encode_df, on = 'offer_id', how = 'left')

    transcript_df = transcript_df.merge(person_encode_df, on = 'person_id', how = 'left')
    transcript_df = transcript_df.merge(offer_encode_df, on = 'offer_id', how = 'left')

    return transcript_df, portfolio_df, profile_df
def pre_model_process(transcript_df, portfolio_df, profile_df):
    '''
    ARGS:
    transcript_df - Transcript Dataframe
    portfolio_df  - Portfolio Dataframe
    profile_df    - Profile Dataframe

    RETURNS:
    df            - Dataframe
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Gets the to-be-used Dataframe and processes it so that its ready for model
    use in computing NaN values

    '''
    part_df = transcript_df.merge(profile_df, on= ['person_id', 'person_index'], how = 'outer')
    model_df = part_df.merge(portfolio_df, on= ['offer_id', 'offer_index'], how = 'outer')

    model_df = process_df(model_df)

    #prep for modelling
    model_df.age.replace(118, np.nan, inplace = True)
    model_df['ordinal_ms'] = model_df['membership_start'].map(datetime.toordinal).to_frame()
    model_df.drop(['membership_start'], axis = 1, inplace = True)
    model_df = pd.get_dummies(model_df, columns = ['event'], drop_first = True)

    return model_df

def model_scoring(model, X_test, y_test, y_pred, skfold = True):
    print(f"Validating {model}")

    results_HV = model.score(X_test, y_test)
    kfold = KFold(n_splits=10, random_state=42, shuffle = True)
    print("Holdout Validation Accuracy: %.2f%%" % (results_HV.mean()*100.0))

    results_kfold = cross_val_score(model, X_test, y_pred, cv=kfold)
    print("KFolds Cross-Validation Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
    if skfold == True:
        skfold = StratifiedKFold(n_splits=3, random_state=42, shuffle = True)
        results_skfold = cross_val_score(model, X, y, cv=skfold)
        print("Stratified K-fold Cross-Validation Accuracy: %.2f%%" % (results_skfold.mean()*100.0))

def create_age_model(df, filename = 'age_pred.sav', rs = 42, score = False):
    '''
    ARGS:
    df         - DataFrame
    filename   - Disired filename for model
    rs         - int or RandomState instance, default = 42
                 Controls the shuffling applied to the data
    score      - Performs Holdout Validation, KFolds Cross-Validation and
                 Stratified K-fold Cross-Validation accuracy checks on the
                 model. Prints results. (default = False)
    RETURNS:
    df         - age_model (for age prediction)
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Builds a model to predict age from and saves it as 'filename' for use later
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Accuracies with Age using RandomForestRegressor:
    Holdout Validation Accuracy: 69.95%
    KFolds Cross-Validation Accuracy: 43.79%

    Accuracies with Age Brackets using RandomForestClassifier:
    Holdout Validation Accuracy: 44.84%
    KFolds Cross-Validation Accuracy: 43.63%
    '''

    X_data = df[df['age'].notna()].copy()
    X_data = X_data.dropna(axis = 1)
    X_data = X_data[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()
    X = X_data
    y = df[df['age'].notna()]['age']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rs)

    age_model = RandomForestRegressor(random_state = rs)
    age_model.fit(X_train,y_train)
    pickle.dump(age_model, open(filename, 'wb'))
    y_pred = age_model.predict(X_test)

    if score == True:
        model_scoring(age_model, X_test, y_test, y_pred, skfold = False)
    return age_model

def create_income_model(df, filename = 'income_pred.sav', rs = 42, score = False):
    '''
    ARGS:
    df          - DataFrame
    filename    - Filename of model
    rs          - int or RandomState instance, default = 42
                  Controls the shuffling applied to the data before applying the
                  split in traintestsplit().
    score       - Performs Holdout Validation, KFolds Cross-Validation and
                  Stratified K-fold Cross-Validation accuracy checks on the
                  model. Prints results. (default = False)

    RETURNS:
    df         - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Impliments income_model for predicting missing ages in dataframe and returns
    model for future use.
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Predicting exact values with such limited data was proving pretty inaccurate.

    The most successful (so far) regression produced the following results:
    (using RandomForestRegressor, RandomState = 42)
    Holdout Validation Accuracy: 52.65%
    KFolds Cross-Validation Accuracy: 46.62%
    Stratified K-fold Cross-Validation Accuracy: 39.56%

    Placing the income into 3 bins for Low Income, Middle Income and High Income
    and using a classifier prediction accuracy increases:
    (using RandomForestClassifier, RandomState = 42)
    Holdout Validation Accuracy: 67.97%
    KFolds Cross-Validation Accuracy: 77.86%
    Stratified K-fold Cross-Validation Accuracy: 76.65%
    '''
    bins = [10000,50000,90000,130000]
    labels = ['low','medium','high']
    df['income_bracket'] = pd.cut(df['income'], bins=bins, labels=labels, right=False)

    X_data = df[df['income_bracket'].notna()].copy()
    X_data = X_data.dropna(axis = 1)
    X_data = X_data[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()
    X = X_data
    y = df[df['income_bracket'].notna()]['income_bracket']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rs)

    income_model = RandomForestClassifier(random_state = rs)
    income_model.fit(X_train,y_train)
    pickle.dump(income_model, open('income_pred.sav', 'wb'))
    y_pred = income_model.predict(X_test)
    if score == True:
        model_scoring(income_model, X_test, y_test, y_pred)

    return income_model

def create_gender_model(df, filename = 'gender_pred.sav', rs = 42, score = False):
    '''
    ARGS:
    df         - DataFrame
    filename   - Disired filename for model
    rs         - int or RandomState instance, default = 42
                 Controls the shuffling applied to the data
    score      - Performs Holdout Validation, KFolds Cross-Validation and
                 Stratified K-fold Cross-Validation accuracy checks on the
                 model. Prints results. (default = False)
    RETURNS:
    df         - age_model (for age prediction)
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Builds a model to predict age from and saves it as 'filename' for use later
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Current Model Accuracy:
    Holdout Validation Accuracy: 70.19%
    KFolds Cross-Validation Accuracy: 74.93%
    Stratified K-fold Cross-Validation Accuracy: 74.20%
    '''

    X_data = df[df['gender'].notna()].copy()
    X_data = X_data.dropna(axis = 1)
    X_data = X_data[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()
    X = X_data

    y_data = df[df['gender'].notna()]['gender']
    y = y_data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rs)

    gender_model = RandomForestClassifier(random_state = rs)
    gender_model.fit(X_train,y_train)
    y_pred = gender_model.predict(X_test)
    pickle.dump(gender_model, open(filename, 'wb'))

    if score == True:
        model_scoring(gender_model, X_test, y_test, y_pred)
    return gender_model

def fill_age_nan(df, filename = 'age_pred.sav', rs = 42, build_model = False, score_model = False):
    '''
    ARGS:
    df          - DataFrame
    filename    - Filename of model
    rs          - int or RandomState instance, default = 42
                  Controls the shuffling applied to the data before applying the
                  split in traintestsplit().
    build_model - When False, function uses saved model. (default = False)
                  When True, function rebuilds model.
    score_model - Performs Holdout Validation, KFolds Cross-Validation and
                  Stratified K-fold Cross-Validation accuracy checks on the
                  model. Prints results. (default = False)
    RETURNS:
    df          - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Impliments age_model for predicting missing ages in dataframe and returns
    DataFrame with added column of predicted ages.
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    '''
    if build_model == True:
        age_model = create_age_model(df, score = score_model)
    else:
        age_model = pickle.load(open(filename, 'rb'))

    pred_df = df[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()

    df['pred_age'] = age_model.predict(pred_df).round()
    df['age'].fillna(df['pred_age'], inplace = True)
    df = age_brackets(df)

    return df
def fill_income_nan(df, filename = 'income_pred.sav', rs = 42, build_model = False, score_model = False):
    '''
    ARGS:
    df          - DataFrame
    filename    - Filename of model
    rs          - int or RandomState instance, default = 42
                  Controls the shuffling applied to the data before applying the
                  split in traintestsplit().
    build_model - When False, function uses saved model. (default = False)
                  When True, function rebuilds model.
    score_model - Performs Holdout Validation, KFolds Cross-Validation and
                  Stratified K-fold Cross-Validation accuracy checks on the
                  model. Prints results. (default = False)

    RETURNS:
    df         - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Impliments income_model for predicting missing ages in dataframe and returns
    DataFrame with added column of predicted ages.
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    '''

    if build_model == True:
        income_model = create_income_model(df, score = score_model)
    else:
        income_model = pickle.load(open(filename, 'rb'))

    pred_df = df[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()

    df['pred_income'] = income_model.predict(pred_df)
    df = income_brackets(df)
    df['income_bracket'].fillna(df['pred_income'], inplace = True)

    return df
def fill_gender_nan(df, filename = 'gender_pred.sav', rs = 42, build_model = False, score_model = False):
    '''
    ARGS:
    df          - DataFrame
    filename    - Filename of model
    rs          - int or RandomState instance, default = 42
                  Controls the shuffling applied to the data before applying the
                  split in traintestsplit().
    build_model - When False, function uses saved model. (default = False)
                  When True, function rebuilds model.
    score_model - Performs Holdout Validation, KFolds Cross-Validation and
                  Stratified K-fold Cross-Validation accuracy checks on the
                  model. Prints results. (default = False)
    RETURNS:
    df         - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Impliments gender_model for predicting missing genders in dataframe and returns
    DataFrame with added column of predicted genders.
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Current Model Accuracy:
    Holdout Validation Accuracy: 70.19%
    KFolds Cross-Validation Accuracy: 74.93%
    Stratified K-fold Cross-Validation Accuracy: 74.20%
    '''
    if build_model == True:
        gender_model = create_gender_model(df, score = score_model)
    else:
        gender_model = pickle.load(open(filename, 'rb'))

    pred_df = df[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()
    df['pred_gender'] = gender_model.predict(pred_df)
    df['gender'].fillna(df['pred_gender'], inplace = True)

    return df

def clean_data(build_model = True, rs = 42, score = False, compute_nans = True):
    '''
    ARGS:
    build_models     - Choose whether to rebuild models using functions or load
                       from saved models.
    rs(random_state) - int or RandomState instance, default = 42. Controls the
                       shuffling applied to the data
    score            - Performs Holdout Validation, KFolds Cross-Validation and
                       Stratified K-fold Cross-Validation accuracy checks on the
                       model. Prints results. (default = False)
    compute_nans     - If True (default), missing values for Age, Gender and
                       income are computed.
                       If False, NaN values are dropped.


    RETURNS:
    transcript_df    - Transcript Dataframe
    portfolio_df     - Portfolio Dataframe
    profile_df       - Profile Dataframe

    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Cleans base data and brings it all together into one DataFrame.

    Income, Age, and Gender NaN values are computed rather than dropped (for now)
    as they represent over 10% of the data set.

    The other NaN values can be easily filled with 0
    '''
    portfolio_df_dirty = pd.read_json('data/portfolio.json', lines = True)
    profile_df_dirty = pd.read_json('data/profile.json', lines = True)
    transcript_df_dirty = pd.read_json('data/transcript.json', lines = True)

    # Clean data
    profile_df = clean_profile_df(profile_df_dirty)
    transcript_df = clean_transcript_df(transcript_df_dirty)
    portfolio_df = clean_portfolio_df(portfolio_df_dirty)

    transcript_df, portfolio_df, profile_df  = id_simpify(transcript_df, portfolio_df, profile_df)
    full_df = pre_model_process(transcript_df, portfolio_df, profile_df)

    # Offer type
    full_df['offer_type'].fillna('transaction', inplace = True)

    #feature list - to be modelled upon
    f_list = ['days_elapsed', 'person_index', 'offer_index', 'ordinal_ms',
    'event_offer received','event_transaction']

    for f in f_list:
        full_df[f].fillna(0, inplace = True)

    if compute_nans == True:
        if build_model == True:
            print('Building Models')
            # merge DataFrames
            full_df = fill_age_nan(full_df, rs, build_model = True, score_model = score)
            print('Finished Age')
            full_df = fill_income_nan(full_df, rs, build_model = True, score_model = score)
            print('Finished Income')
            full_df = fill_gender_nan(full_df, rs, build_model = True, score_model = score)
            print('Finished Gender')
        else:
            # merge DataFrames
            full_df = fill_age_nan(full_df)
            # full_df = fill_age_nan(full_df, rs, build_model = True, score_model = True)
            print('Finished Age')
            full_df = fill_income_nan(full_df)
            print('Finished Income')
            full_df = fill_gender_nan(full_df)
            print('Finished Gender')
    else:
        full_df = full_df[full_df['gender'].notna()].copy()
    # The Rest
    for x in full_df.columns:
        try:
            full_df[x].fillna(0, inplace = True)
        except:
            continue
    print('Clean Completed')
    return full_df
#
# clean_df = clean_data(compute_nans = True)
# clean_df.to_pickle('./clean_data.pkl')
