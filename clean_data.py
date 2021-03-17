import pandas as pd
from datetime import datetime
from process_data import *
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


def clean_data():
    '''
    ARGS: None

    RETURNS:
    full_df - DataFrame

    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Cleans base data and brings it all together into one DataFrame.

    Income, Age, and Gender NaN values are dropped as they are the same rows and
    although they represent around 10% of the data set they reduce modelling
    accuracy by a similar amount.

    After extensive testing its deemed most appropriate to exclude this data for now.
    This will remain the case until either more testing can be done or a better
    estimate of missing data is found.

    The other NaN values can be easily filled with 0

    For previous testing models see /redundant_functions.py
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
    print('Loaded Up')
    # Offer type
    full_df['offer_type'].fillna('transaction', inplace = True)

    f_list = ['days_elapsed', 'person_index', 'offer_index', 'ordinal_ms',
    'event_offer received','event_transaction']

    for f in f_list:
        full_df[f].fillna(0, inplace = True)

    full_df = full_df[full_df['gender'].notna()].copy()
    # The Rest
    for x in full_df.columns:
        try:
            full_df[x].fillna(0, inplace = True)
        except:
            continue
    # model_df['ordinal_ms'] = model_df['membership_start'].map(datetime.toordinal).to_frame()
    # model_df.drop(['membership_start'], axis = 1, inplace = True)

    print('Clean Completed')
    return full_df



# clean_df = clean_data(compute_nans = True)
# clean_df.to_pickle('./data/clean_data.pkl')
