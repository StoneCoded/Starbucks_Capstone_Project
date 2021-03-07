import pandas as pd
from datetime import datetime
from clean_data import *
from process_data import *
import matplotlib.pyplot as plt

from scipy.stats.stats import pearsonr

# Get Data and Process it
transcript_df, portfolio_df, profile_df = clean_data()
profile_df.groupby('age_bracket')['age'].count().plot(kind = 'bar')
profile_df.groupby('gender')['age'].count().plot(kind = 'bar')

part_over_df = transcript_df.merge(profile_df, on= ['person_id', 'person_index'], how = 'outer')
overview_df = part_over_df.merge(portfolio_df, on= ['offer_id', 'offer_index'], how = 'outer')

overview_df = process_df(overview_df)

overview_df[overview_df['event'] == 'offer completed']


overview_df[overview_df['person_index'] == 1]
overview_df[overview_df['person_index'] == 1]
overview_df.amount.isna().sum()
overview_df['event']





# total offers, including failed ones
overview_df['event_s'] = overview_df.event.apply(lambda x: 'offer' if ((x == 'offer received') | (x == 'offer completed')) else 'NaN')
overview_df['profit'] = (overview_df.amount - overview_df.reward)

# break down events
event_df = overview_df.groupby(['days_elapsed', 'event'])['person_index'].count().to_frame().copy()
all_off = overview_df.groupby(['days_elapsed', 'event_s'])['person_index'].count().to_frame().copy()

# completed offers
off_comp_df = overview_df[overview_df['event'] == 'offer completed'].copy()
off_com_type = off_comp_df.groupby(['days_elapsed', 'offer_type'])['person_index'].count().to_frame()

## Time based values
# Transactions, completed offers and total offers over time
all_trans = event_df.iloc[(event_df.index.get_level_values('event') == 'transaction')].copy()
comp_off = event_df.iloc[(event_df.index.get_level_values('event') == 'offer completed')].copy()
all_off = all_off.iloc[(all_off.index.get_level_values('event_s') == 'offer')].copy()

# break down of completed offers types
off_com_bogo = off_com_type.iloc[off_com_type.index.get_level_values('offer_type') == 'bogo']
off_com_info = off_com_type.iloc[off_com_type.index.get_level_values('offer_type') == 'informational']
off_com_disc = off_com_type.iloc[off_com_type.index.get_level_values('offer_type') == 'discount']

# money based values
reward_df = overview_df.groupby(['days_elapsed'])['reward'].sum().to_frame().copy()
revenue_df = overview_df.groupby(['days_elapsed'])['amount'].sum().to_frame().copy()
profit_df = overview_df.groupby(['days_elapsed'])['profit'].sum().to_frame().copy()

# Cumulative totals
revenue_df['total_rev'] = revenue_df.amount.cumsum()
all_off['tot_offers'] = all_off.person_index.cumsum()
all_trans['tot_trans'] = all_trans.person_index.cumsum()
comp_off['tot_comp'] = comp_off.person_index.cumsum()

# Plot n Save
fig = plt.figure()

for frame in [all_trans, comp_off]:
    x = frame.iloc[:,0].values
    normalized = (x-min(x))/(max(x)-min(x))
    plot = plt.plot(frame.index.get_level_values('days_elapsed').values, normalized)
    plt.xlabel("Days Elapsed")
    plt.ylabel("")
    plt.title("Total Transactions vs Total Offers")
    plt.legend(('Total Transactions', 'All Offers'))

plt.savefig(f"output{7}.png", dpi=300, bbox_inches = "tight")

plt.show()
