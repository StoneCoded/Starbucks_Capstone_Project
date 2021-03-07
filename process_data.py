import pandas as pd
import numpy as np

def trans_comp_sort(df):
    '''
    ARGS: df - DataFrame

    RETURNS: df - DataFrame

    ––––––––––––––––––––––––––––––––––––––––––––––––––
    For each Offer Completed there is a corresponding Transaction with exactly
    the same data save the amount. This will muddle the actual transaction data.

    trans_comp_sort takes the 'amount', inputs it into the Offer Completed
    cell and removes the Transaction.
    '''
    #Initial Variables, (-1 as it doesn't exist in loop)
    drop_list = []
    #time variables
    last_tran_time = -1
    last_comp_time = -1
    #index variables
    t_idx = -1
    c_idx = -1
    #person_id variables
    p_id_t = -1
    p_id_c = -1

    #sort_values
    df = df.sort_values(by = ['person_index','days_elapsed','offer_index'])
    df.reset_index(drop = True, inplace = True)

    for m in range(len(df)):

        if df.loc[m, 'event'] == 'transaction':
            last_tran_time = df.loc[m, 'days_elapsed']
            t_idx = m
            p_id_t = df.loc[m, 'person_id']

            if (last_tran_time == last_comp_time) & (p_id_t == p_id_c):
                df.at[c_idx, 'amount'] = df.loc[t_idx, 'amount']
                drop_list.append(t_idx)
            else:
                continue

        elif df.loc[m, 'event'] == 'offer completed':
            last_comp_time = df.loc[m, 'days_elapsed']
            c_idx = m
            p_id_c = df.loc[m, 'person_id']

            if (last_tran_time == last_comp_time) & (p_id_t == p_id_c):
                df.at[c_idx, 'amount'] = df.loc[t_idx, 'amount']
                drop_list.append(t_idx)
            else:
                continue

    if len(drop_list) > 0:
        df.drop(df.index[np.unique(drop_list)], inplace = True)
        df.reset_index(drop = True, inplace = True)
    return df
def received_sort(df):
    '''
    ARGS: df - DataFrame

    RETURNS: df - DataFrame

    ––––––––––––––––––––––––––––––––––––––––––––––––––
    A completed offer is generally determined a success. By filtering out
    the offers received that went onto be successful we can see the unsuccessful
    offers and learn from them.

    received_sort removes 'Offer Received' if there is a corresponding Offer
    Completed.
    '''
    #Initial Variables, (-1 as it doesn't exist in loop)
    drop_list = []

    #time variables
    last_rec_time = -1
    last_comp_time = -1

    #index variables
    r_idx = -1
    c_idx = -1
    o_idx = -1
    o_idx_2 = -1

    #person_id variables
    p_id_r = -1
    p_id_c = -1

    #time duration variables
    r_dur = -1
    time_diff = -1

    df = df.sort_values(by = ['person_index','offer_index','days_elapsed'])
    df.reset_index(drop = True, inplace = True)
    for m in range(len(df)):

        if df.loc[m, 'event'] == 'offer received':
            last_rec_time = df.loc[m, 'days_elapsed']
            p_id_r = df.loc[m, 'person_id']
            o_idx = df.loc[m, 'offer_id']

            r_idx = m
            r_dur = df.loc[m, 'duration']

        elif df.loc[m, 'event'] == 'offer completed':
            last_comp_time = df.loc[m, 'days_elapsed']
            p_id_c = df.loc[m, 'person_id']
            o_idx_2 = df.loc[m, 'offer_id']

            time_elap = last_comp_time - last_rec_time
            time_diff = r_dur - time_elap

            if (time_diff > 0) & (p_id_r == p_id_c) & (o_idx == o_idx_2):
                drop_list.append(r_idx)
            else:
                continue

    if len(drop_list) > 0:
        df.drop(df.index[np.unique(drop_list)], inplace = True)
        df.reset_index(drop = True, inplace = True)
    return df
def viewed_sort(df):
    '''
    ARGS: df - DataFrame

    RETURNS: df - DataFrame
    –––––––––––––––––––––––––––––––––––
    Drops 'offer viewed' from DataFrame

    '''
    df = df[df['event'] != 'offer viewed']
    df.reset_index(drop = True, inplace = True)
    return df
def informational_sort(df):
    '''
    ARGS: df - DataFrame

    RETURNS: df - DataFrame

    ––––––––––––––––––––––––––––––––––––––––––––––––––
    If a purchase is made within a duration of an informational offer it is
    deamed a success.

    informational_sort removes 'transactions' if it is deamed successful and
    replaces the Offer Received as Offer Completed with all corresponding info.
    '''
    #Initial Variables, (-1 as it doesn't exist in loop)
    success_list = []
    drop_list = []
    last_info_time = -1
    last_trans_time = -1

    i_idx = -1
    p_id_i = -1
    p_id_tt = -1
    i_dur = -1
    time_diff = -1
    #sort by time
    df = df.sort_values(by = ['person_index', 'days_elapsed', 'offer_index'])
    df.reset_index(drop = True, inplace = True)
    for m in range(len(df)):
        #find each informational offer
        if df.loc[m, 'offer_type'] == 'informational':
            #update variables
            last_info_time = df.loc[m, 'days_elapsed']
            p_id_i = df.loc[m, 'person_id']

            i_idx = m
            i_dur = df.loc[m, 'duration']

        #find each transaction
        elif df.loc[m, 'event'] == 'transaction':
            last_trans_time = df.loc[m, 'days_elapsed']
            p_id_t = df.loc[m, 'person_id']

            time_elap = last_trans_time - last_info_time
            time_diff = i_dur - time_elap
            #check if user is the same
            if (time_diff >= 0) & (p_id_i == p_id_t):
                #check if transaction is linked with offer
                if (i_idx not in success_list):
                    #update idx lists
                    success_list.append(i_idx)
                    drop_list.append(m)

                    #update Offer Received
                    df.at[i_idx, 'event'] = 'offer completed'
                    df.at[i_idx, 'amount'] = df.loc[m, 'amount']
                    df.at[i_idx, 'days_elapsed'] = df.loc[m, 'days_elapsed']
                    df.at[i_idx, 'success'] = 1

    if len(drop_list) > 0:
        #drop transactions
        df.drop(df.index[np.unique(drop_list)], inplace = True)
        df.reset_index(drop = True, inplace = True)
    return df

def process_df(df, trans_comp = True, rec_sort = True, view_sort = True, info_sort = True, success = True):

    '''
    ARGS:
    df         - DataFrame
    trans_comp - Takes the 'amount' value, inputs it into the Offer Completed
                 cell and removes the Transaction.
    rec_sort   - Removes 'Offer Received' if there is a corresponding Offer
                 Completed.
    view_sort  - Drops 'offer viewed' from DataFrame, resets index
    info_sort  - Removes 'transactions' if it is deamed successful and replaces
                 the Offer Received as Offer Completed with all corresponding info.
    success    - Adds success column based of if a transaction was made or an
                 offer completed.
    RETURNS:
    df         - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Default for all funcion ARGS is True.

    Takes cleaned DataFrame and processes it further but leaves NaNs and category
    data for a later processing thus not limiting choice.

    '''
    if trans_comp == True:
        df = trans_comp_sort(df)
    if trans_comp == True:
        df = received_sort(df)
    if trans_comp == True:
        df = viewed_sort(df)
    if trans_comp == True:
        df = informational_sort(df)
    if success == True:
        df['success'] = df.event.apply(lambda x: 1 if (x == 'offer completed') | (x == 'transaction') else 0)

    return df





# df.reward.fillna(0, inplace = True)
df.offer_type.fillna('no offer', inplace = True)
# df.email.fillna(0, inplace = True)

for x in df.columns:

    try:
        df[x].fillna(0, inplace = True)
    except:
        continue

# sub.head()

# df['sum'] = df.person_index.apply(lambda x: 1 if x != 'NaN' else 0)
# sub = df.copy()
# df_events = pd.get_dummies(df.event.apply(pd.Series).stack()).sum(level=0)
# sub = sub.join(df_events)
# sub = sub.drop('event', axis = 1)
#
# df_gender = pd.get_dummies(df.gender.apply(pd.Series).stack()).sum(level=0)
# sub = sub.join(df_gender)
# sub = sub.drop('gender', axis = 1)
# sub
# df_off_type = pd.get_dummies(df.offer_type.apply(pd.Series).stack()).sum(level=0)
# sub = sub.join(df_off_type)
# sub = sub.drop('offer_type', axis = 1)
# df_off_type.columns = ['none', 'bogo', 'discount', 'informational']
# sub = sub.drop('person_id', axis = 1)
# sub = sub.drop('offer_id', axis = 1)
# sub = sub.drop('membership_start', axis = 1)
#
# df_a_b = pd.get_dummies(df.age_bracket.apply(pd.Series).stack()).sum(level=0)
# sub = sub.join(df_a_b)
# sub = sub.drop('age_bracket', axis = 1)
# sub
#
# sub.person_index.unique().tolist()
df['profit'] = (sub.amount - sub.reward)
# sub
# sub[sub.person_index == 98]
