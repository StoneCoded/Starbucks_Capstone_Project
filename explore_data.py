import pandas as pd
from datetime import datetime
from clean_data import *
from process_data import *
from offer_rec_functions import *
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.stats import pearsonr

'''
Fair Warning.

This file is a minefield and impossible to read unless you're me. :)


'''

# Get Data and Process it
overview_df = load_data()
# test = pd.read_pickle('amount_age_test.pickle')
amount_pred = pd.read_pickle('./data/amount_prediction.pkl')
amount_pred_full = pd.read_pickle('./data/amount_prediction_full.pkl')
#
# outliers = off_comp_df[(off_comp_df.amount < lower) & (off_comp_df.amount > upper)].copy()
# len(outliers)
#
# outli
# q25 = off_comp_df.amount.describe()[4]
# q75 = off_comp_df.amount.describe()[6]
# iqr = (q75 - q25)
# cut_off = iqr * 1.5
# lower, upper = q25 - cut_off, q75 + cut_off
# o1 = amount_pred_full.iloc[0::10,:].amount
# o2 = amount_pred_full.iloc[1::10,:].amount.values
# o3 = amount_pred_full.iloc[2::10,:].amount.values
# o4 = amount_pred_full.iloc[3::10,:].amount.values
# o5 = amount_pred_full.iloc[4::10,:].amount.values
# o6 = amount_pred_full.iloc[5::10,:].amount.values
# o7 = amount_pred_full.iloc[6::10,:].amount.values
# o8 = amount_pred_full.iloc[7::10,:].amount.values
# o9 = amount_pred_full.iloc[8::10,:].amount.values
# o10 = amount_pred_full.iloc[9::10,:].amount.values
overview_df.groupby('income')['amount'].describe().iloc[:,4]
overview_df.groupby('income')['amount'].describe().iloc[:,6]




overview_df.amount.describe()
amount_pred_full.amount.describe()
def graphic(target):

    mins = amount_pred_full.groupby(target)['amount'].min()
    maxs = amount_pred_full.groupby(target)['amount'].max()
    avg = amount_pred_full.groupby(target)['amount'].mean().values
    index = amount_pred_full.groupby(target)['amount'].mean().index

    known_avg = overview_df.groupby(target)['amount'].mean().values
    known_max = overview_df.groupby(target)['amount'].max().values
    known_min = overview_df.groupby(target)['amount'].min().values
    known_index = overview_df.groupby(target)['amount'].std().index

    plt.plot(known_index, known_avg, label = 'True Average')
    plt.plot(index, avg, color = 'red', alpha = 0.6, label = 'Predicted Average')
    plt.savefig(f"output_{target}1.png", dpi=300, bbox_inches = "tight")
    plt.title(f'True vs Predicted Average Transaction Based on {target.capitalize()}')
    plt.xlabel(target.capitalize())
    plt.ylabel('Transaction Amount ($)')
    plt.legend()
    plt.show()

    plt.scatter(known_index, known_max, color = '#1f77b4', label = 'True Limits')
    plt.scatter(index, maxs, color = 'red', alpha = 0.6, label = 'Predicited Limits')
    plt.scatter(known_index, known_min, color = '#1f77b4')
    plt.scatter(index, mins, color = 'red', alpha = 0.6)
    plt.savefig(f"output_{target}2.png", dpi=300, bbox_inches = "tight")
    plt.title(f'True vs Predicted Transaction Limits Based on {target.capitalize()}')
    plt.xlabel(target.capitalize())
    plt.ylabel('Transaction Amount ($)')
    plt.legend()
    plt.show()

graphic('age')

plt.hist(overview_df.amount.values)

known = overview_df.groupby('income')['amount'].mean().values
incomes = overview_df.groupby('income')['amount'].sum().index

pred = amount_pred.groupby('income')['amount'].mean().values
pred_full = amount_pred_full.groupby('income')['amount'].mean().values

plt.plot(incomes, known)
plt.scatter(incomes, pred_full, color = 'red', alpha = 0.6)
# plt.scatter(incomes, pred, color = 'red', alpha = 0.6)

member_length_sum = overview_df.groupby('member_length')['amount'].mean().values
members_lens = member_length_sum = overview_df.groupby('member_length')['amount'].sum().index


pred_member_length_sum = amount_pred.groupby('member_length')['amount'].mean().values
pred_lens = amount_pred.groupby('member_length')['amount'].sum().index

known = member_length_sum/member_length_count
pred = pred_member_length_sum/pred_member_length_count

normalized_known = (known - min(known)) / (max(known) - min(known))
normalized_pred = (pred - min(pred)) / (max(pred) - min(pred))




plt.scatter(members_lens, member_length_sum, alpha = 0.6)
plt.scatter(pred_lens, pred_member_length_sum, alpha = 0.4, color = 'red')
plt.ylim(0,25)



fig, ax = plt.subplots()
labels = ['email', 'mobile', 'social', 'web']
x = range(len(off_comp_df[['email', 'mobile', 'social', 'web']].sum().values))
ax.bar(x, (off_comp_df[['email', 'mobile', 'social', 'web']].sum().values/off_rec_df[['email', 'mobile', 'social', 'web']].sum().values)*100)
ax.set_ylim(50,65)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Percent Completion')
ax.set_title('Offer Medium Success')
plt.savefig(f"output{'_med'}.png", dpi=300, bbox_inches = "tight")
# off_comp_df[['email', 'mobile', 'social', 'web']].sum().plot(kind = 'bar', title = 'Medium Success Totals')

off_comp_df.shape[0]/off_rec_df.shape[0]

off_rec_gen = off_rec_df.groupby(['gender', 'offer_type'])['person_index'].count().to_frame()
off_comp_gen = off_comp_df.groupby(['gender', 'offer_type'])['person_index'].count().to_frame()


off_com_type = off_comp_df.groupby(['days_elapsed', 'offer_type'])['person_index'].count().to_frame()
off_com_age = off_rec_df.groupby(['age', 'offer_type'])['person_index'].count().to_frame()
off_com_income = off_rec_df.groupby(['income', 'offer_type'])['person_index'].count().to_frame()


## Time based values
# Transactions, completed offers and total offers over time
all_trans = event_df.iloc[(event_df.index.get_level_values('event') == 'transaction')].copy()
comp_off = event_df.iloc[(event_df.index.get_level_values('event') == 'offer completed')].copy()
all_off = all_off.iloc[(all_off.index.get_level_values('event_s') == 'offer')].copy()

rec_gen_bogo = off_rec_gen.iloc[off_rec_gen.index.get_level_values('offer_type') == 'bogo']
rec_gen_info = off_rec_gen.iloc[off_rec_gen.index.get_level_values('offer_type') == 'informational']
rec_gen_disc = off_rec_gen.iloc[off_rec_gen.index.get_level_values('offer_type') == 'discount']

comp_gen_bogo = off_comp_gen.iloc[off_comp_gen.index.get_level_values('offer_type') == 'bogo']
comp_gen_info = off_comp_gen.iloc[off_comp_gen.index.get_level_values('offer_type') == 'informational']
comp_gen_disc = off_comp_gen.iloc[off_comp_gen.index.get_level_values('offer_type') == 'discount']
comp_gen_trans = off_comp_gen.iloc[off_comp_gen.index.get_level_values('offer_type') == 'transaction']


# break down of completed offers types
off_com_bogo = off_com_type.iloc[off_com_type.index.get_level_values('offer_type') == 'bogo']
off_com_info = off_com_type.iloc[off_com_type.index.get_level_values('offer_type') == 'informational']
off_com_disc = off_com_type.iloc[off_com_type.index.get_level_values('offer_type') == 'discount']

comp_age_bogo = off_com_age.iloc[off_com_age.index.get_level_values('offer_type') == 'bogo']
comp_age_info = off_com_age.iloc[off_com_age.index.get_level_values('offer_type') == 'informational']
comp_age_disc = off_com_age.iloc[off_com_age.index.get_level_values('offer_type') == 'discount']

comp_income_bogo = off_com_income.iloc[off_com_income.index.get_level_values('offer_type') == 'bogo']
comp_income_info = off_com_income.iloc[off_com_income.index.get_level_values('offer_type') == 'informational']
comp_income_disc = off_com_income.iloc[off_com_income.index.get_level_values('offer_type') == 'discount']


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
    plot = plt.scatter(frame.index.get_level_values('days_elapsed').values, normalized)
    plt.xlabel("Days Elapsed")
    plt.ylabel("")
    plt.title("Total Transactions vs Total Offers")
    plt.legend(('Total Transactions', 'All Offers'))

plt.savefig(f"output{7}.png", dpi=300, bbox_inches = "tight")
plt.show()


fig, ax = plt.subplots()

bx = comp_gen_bogo.iloc[:,0].values
# bx = rec_gen_bogo.iloc[:,0].values
bcr = comp_gen_bogo.iloc[:,0].values/rec_gen_bogo.iloc[:,0].values
# normalized_bogo = (bx-min(bx))/(max(bx)-min(bx))

ix = comp_gen_info.iloc[:,0].values
# ix = rec_gen_info.iloc[:,0].values
# normalized_info = (ix-min(ix))/(max(ix)-min(ix))
icr = comp_gen_info.iloc[:,0].values/rec_gen_info.iloc[:,0].values

dx = comp_gen_disc.iloc[:,0].values
# dx = rec_gen_disc.iloc[:,0].values
# normalized_disc = (dx-min(dx))/(max(dx)-min(dx))
dcr = comp_gen_disc.iloc[:,0].values/rec_gen_disc.iloc[:,0].values
# tx = comp_gen_trans.iloc[:,0].values

labels = ['Female', 'Male', 'Other']
w = np.arange(len(labels))
width = 0.2
ax.bar(w - width/2, bcr, width = width/2, label = 'Informational')
ax.bar(w + width/2, icr, width = width/2, label = 'BOGO')
ax.bar(w, dcr, width = width/2, label = 'Discount')
# ax.bar(w + width, tx, width = width/2, label = 'Transaction')
ax.set_xticks(w)
ax.set_xticklabels(labels)
# ax.legend()

# plt.xlabel("Gender")
plt.ylabel("Offer Completion Rate")
plt.title("Offer Completion by Gender")
plt.savefig(f"output{'_gen_comp_rate'}.png", dpi=300, bbox_inches = "tight")
# plt.legend(('BOGO', 'Informational', 'Discount'))


fig = plt.figure()
for frame in [comp_gen_bogo, comp_gen_info, comp_gen_disc]:
    x = frame.iloc[:,0].values
    normalized = (x-min(x))/(max(x)-min(x))
    plot = plt.bar(frame.index.get_level_values('gender').values, x)
    plt.xlabel("Age")
    plt.ylabel("Offers")
    plt.title("Offers Received by Age")
    plt.legend(('BOGO', 'Informational', 'Discount'))
    # plt.xlim(left = 0, right = 20)

plt.savefig(f"output{15}.png", dpi=300, bbox_inches = "tight")
