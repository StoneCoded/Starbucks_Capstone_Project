import pandas as pd
import numpy as np
from datetime import datetime

'''
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
    df = df.dropna().reset_index(drop = True)
    df.columns = ['gender', 'age', 'person_id', 'membership_start','income']

    df['membership_start'] = [datetime.strptime(str(x), '%Y%m%d').\
                        strftime('%m/%d/%Y') for x in df.membership_start]

    df['membership_start'] = pd.to_datetime(df.membership_start)

    bins = [18,25,35,45,55,65,75,85,95,105]
    labels = ['18-24','25-34','35-44','45-54','55-64','65-74','75-84','85-94','95-104']
    df['age_bracket'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    return df
def clean_data():

    portfolio_df_dirty = pd.read_json('data/portfolio.json', lines = True)
    profile_df_dirty = pd.read_json('data/profile.json', lines = True)
    transcript_df_dirty = pd.read_json('data/transcript.json', lines = True)

    # Clean data
    profile_df = clean_profile_df(profile_df_dirty)
    transcript_df = clean_transcript_df(transcript_df_dirty)
    portfolio_df = clean_portfolio_df(portfolio_df_dirty)

    offer_ids = portfolio_df_dirty.id.unique().tolist()
    offer_ids_2 = transcript_df.offer_id.unique().tolist()

    person_ids = profile_df_dirty.id.unique().tolist()
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
