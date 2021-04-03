import pandas as pd
from datetime import datetime
from clean_data import *
from process_data import *
from offer_rec_functions import *
from redundant_functions import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from scipy.stats.stats import pearsonr

'''
Fair Warning.

This file is a minefield and impossible to read unless you're me. :)


'''

# Get Data and Process it
overview_df = load_data()

#THIS
overview_df.groupby('duration')['success'].sum()

success_df = overview_df[overview_df['success'] == 1].copy()

amount_pred = pd.read_pickle('./data/amount_prediction.pkl')
amount_pred_full = pd.read_pickle('./data/amount_prediction_full.pkl')
amount_pred_full = pd.read_pickle('./data/amount_prediction_full_2.pkl')
amount_pred_full = pd.read_pickle('./data/amount_prediction_full_3.pkl')

plt.plot(overview_df.groupby('age')['success'].count().index, overview_df.groupby('age')['success'].count().values)
plt.scatter(amount_pred_full.groupby('age')['success'].count().index, amount_pred_full.groupby('age')['success'].count().values, color = 'orange')


def graphic(target):

    mins = amount_pred_full.groupby(target)['amount'].min()
    maxs = amount_pred_full.groupby(target)['amount'].max()
    avg = amount_pred_full.groupby(target)['amount'].mean().values
    std = amount_pred_full.groupby(target)['amount'].describe().iloc[:,4].values
    std1 = amount_pred_full.groupby(target)['amount'].describe().iloc[:,6].values

    index = amount_pred_full.groupby(target)['amount'].mean().index

    known_avg = overview_df.groupby(target)['amount'].mean().values
    known_std = overview_df.groupby(target)['amount'].describe().iloc[:,4].values
    known_std1 = amount_pred_full.groupby(target)['amount'].describe().iloc[:,6].values

    known_max = overview_df.groupby(target)['amount'].max().values
    known_min = overview_df.groupby(target)['amount'].min().values
    known_index = overview_df.groupby(target)['amount'].describe().iloc[:,4].index

    plt.plot(known_index, known_avg, label = 'True Average')
    plt.plot(index, avg, color = 'red', alpha = 0.5, label = 'Predicted Average')
    plt.savefig(f"output_{target}4.png", dpi=300, bbox_inches = "tight")
    plt.title(f'True vs Predicted Average Transaction Based on {target.capitalize()}')
    plt.xlabel(target.capitalize())
    plt.ylabel('Transaction Amount ($)')
    plt.legend()
    plt.show()

    plt.scatter(known_index, known_std, color = 'black', label = 'True Standard Deviation')
    plt.scatter(known_index, known_std1, color = 'green', label = 'True Standard Deviation')
    plt.scatter(index, std, color = 'red', alpha = 0.5, label = 'Predicted Standard Deviation')
    plt.scatter(index, std1, color = 'yellow', alpha = 0.5, label = 'Predicted Standard Deviation')
    # plt.savefig(f"output_{target}5.png", dpi=300, bbox_inches = "tight")
    plt.title(f'True vs Predicted Standard Deviation Based on {target.capitalize()}')
    plt.xlabel(target.capitalize())
    plt.ylabel('Transaction Amount ($)')
    plt.legend()
    plt.show()

    plt.scatter(known_index, known_max, color = '#1f77b4', label = 'True Limits', alpha = 0.5)
    plt.scatter(index, maxs, color = 'red', alpha = 0.5, label = 'Predicited Limits')
    plt.scatter(known_index, known_min, color = '#1f77b4', alpha = 0.5)
    plt.scatter(index, mins, color = 'red', alpha = 0.5)
    plt.savefig(f"output_{target}6.png", dpi=300, bbox_inches = "tight")
    plt.title(f'True vs Predicted Transaction Limits Based on {target.capitalize()}')
    plt.xlabel(target.capitalize())
    plt.ylabel('Transaction Amount ($)')
    plt.legend()
    plt.show()
graphic('age')
graphic('income')


person_df = overview_df.groupby(['person_id', 'age','income', 'gender', 'member_length']).count().reset_index().iloc[:,:5].copy()

def gen_graphic(file = amount_pred_full):
    f_avg = file.groupby('gender_F')['amount'].mean().values[1]
    m_avg = file.groupby('gender_M')['amount'].mean().values[1]
    o_avg = file.groupby('gender_O')['amount'].mean().values[1]


    over_slice = overview_df[['amount','gender']].copy()
    over_slice = pd.get_dummies(over_slice)

    kf_avg = over_slice.groupby('gender_F')['amount'].mean().values[1]
    km_avg = over_slice.groupby('gender_M')['amount'].mean().values[1]
    ko_avg = over_slice.groupby('gender_O')['amount'].mean().values[1]

    pred_list = [f_avg, m_avg, o_avg]
    k_list = [kf_avg, km_avg, ko_avg]

    fig, ax = plt.subplots()
    width = 0.4
    x = np.arange(len(pred_list))
    ax.bar(x - width/4, pred_list, width = width/2, label = 'Predicted')
    ax.bar(x + width/4, k_list, width = width/2, label = 'True')
    ax.set_xticks(x)
    ax.set_xticklabels(['Female','Male','Other'])
    ax.set_title('Average Transaction')
    ax.set_ylabel('Transaction Amount($)')
    ax.legend()
    plt.savefig(f"output_gender_avg.png", dpi=300, bbox_inches = "tight")
    plt.show()
gen_graphic()

def overview_graph_basic(df):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.tight_layout()
    plt.subplots_adjust(hspace = 0.6)

    ax1.hist(df.age.values)
    ax1.set_title('Age')

    ax2.hist(df.income.values)
    ax2.set_xticks([40000,60000,80000,100000,120000])
    ax2.set_xticklabels(['40k','60k','80k','100k','120k'])
    ax2.set_title('Annual Income')

    gndr_cnt =[df[df.gender == 'M'].shape[0],
               df[df.gender == 'F'].shape[0],
               df[df.gender == 'O'].shape[0]]

    ax3.bar(['Male','Female','Other'], gndr_cnt)
    ax3.set_title('Gender')

    ax4.hist(df.member_length.values)
    ax4.set_xticks([0,365,730,1095, 1460, 1825])
    ax4.set_xticklabels(['0','1','2','3','4', '5'])
    ax4.set_title('Membership Length (Years)')

    plt.savefig(f"overview_demo_full.png", dpi=300, bbox_inches = "tight")
person_df = overview_df.groupby(['person_id', 'age','income', 'gender', 'member_length']).count().reset_index().iloc[:,:5].copy()

overview_graph_basic(person_df)
overview_graph_basic(overview_df)

def overview_graph(df2, df3, df4, labels, filename):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9,6))
    fig.tight_layout()
    plt.subplots_adjust(hspace = 0.3)

    ax1.hist(df2.age.values, alpha = 0.5)
    ax1.set_title('Age')

    ax2.hist(df2.income.values, alpha = 0.5)
    ax2.set_xticks([40000,60000,80000,100000,120000])
    ax2.set_xticklabels(['40k','60k','80k','100k','120k'])
    ax2.set_title('Annual Income')

    gndr_cnt =[df2[df2.gender == 'M'].shape[0],
               df2[df2.gender == 'F'].shape[0],
               df2[df2.gender == 'O'].shape[0]]

    ax3.bar(['Male','Female','Other'], gndr_cnt, alpha = 0.5)
    ax3.set_title('Gender')

    ax4.hist(df2.member_length.values, alpha = 0.5)
    ax4.set_xticks([0,365,730,1095, 1460, 1825])
    ax4.set_xticklabels(['0','1','2','3','4', '5'])
    ax4.set_title('Membership Length (Years)')

    ax1.hist(df3.age.values, alpha = 0.5)
    ax1.set_title('Age')

    ax2.hist(df3.income.values, alpha = 0.5)
    ax2.set_xticks([40000,60000,80000,100000,120000])
    ax2.set_xticklabels(['40k','60k','80k','100k','120k'])
    ax2.set_title('Annual Income')

    gndr_cnt =[df3[df3.gender == 'M'].shape[0],
               df3[df3.gender == 'F'].shape[0],
               df3[df3.gender == 'O'].shape[0]]

    ax3.bar(['Male','Female','Other'], gndr_cnt, alpha = 0.5)
    ax3.set_title('Gender')

    ax4.hist(df3.member_length.values, alpha = 0.5)
    ax4.set_xticks([0,365,730,1095, 1460, 1825])
    ax4.set_xticklabels(['0','1','2','3','4', '5'])
    ax4.set_title('Membership Length (Years)')

    ax1.hist(df4.age.values, alpha = 0.5)
    ax1.set_title('Age')

    ax2.hist(df4.income.values, alpha = 0.5)
    ax2.set_xticks([40000,60000,80000,100000,120000])
    ax2.set_xticklabels(['40k','60k','80k','100k','120k'])
    ax2.set_title('Annual Income')

    gndr_cnt =[df4[df4.gender == 'M'].shape[0],
               df4[df4.gender == 'F'].shape[0],
               df4[df4.gender == 'O'].shape[0]]

    ax3.bar(['Male','Female','Other'], gndr_cnt, alpha = 0.5)
    ax3.set_title('Gender')
    ax3.legend(labels)

    ax4.hist(df4.member_length.values, alpha = 0.5)
    ax4.set_xticks([0,365,730,1095, 1460, 1825])
    ax4.set_xticklabels(['0','1','2','3','4', '5'])
    ax4.set_title('Membership Length (Years)')
    # plt.legend(['Transaction','Successful Offer','Failed Offer'])
    plt.savefig(filename, dpi=300, bbox_inches = "tight")

overview_graph(overview_df[overview_df['event_transaction'] == 1],
               overview_df[overview_df['event_offer completed'] == 1],
               overview_df[overview_df['event_offer received'] == 1],
               ['Transaction','Successful Offer', 'Failed Offer'],
               f"event_compare_demo.png")
overview_graph(overview_df[overview_df['offer_type'] == 'discount'],
               overview_df[overview_df['offer_type'] == 'bogo'],
               overview_df[overview_df['offer_type'] == 'informational'],
               ['Discount','BOGO', 'Informational'],
               f"offer_compare_demo.png")

success = overview_df[overview_df['success'] == 1]
overview_graph(success[success['offer_type'] == 'discount'],
               success[success['offer_type'] == 'bogo'],
               success[success['offer_type'] == 'informational'],
               ['Discount','BOGO', 'Informational'],
               f"Successful_offer_compare_demo.png")


comptrans = overview_df[(overview_df['event_offer completed'] == 1) | (overview_df['event_transaction'] == 1)].copy()
alloff = overview_df[(overview_df['event_offer completed'] == 1) | (overview_df['event_offer received'] == 1)].copy()

(comptrans.groupby(['days_elapsed'])['person_index'].count().values)
plt.plot(comptrans.groupby(['days_elapsed'])['person_index'].count().index, comptrans.groupby(['days_elapsed'])['person_index'].count().values)
plt.plot(alloff.groupby(['days_elapsed'])['person_index'].count().index, alloff.groupby(['days_elapsed'])['person_index'].count().values)

def avg_spend_age_income(df):
    '''
    Plots Average spend for Age and Income
    '''
    fig, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(9,6), sharey = True)
    ax1.plot(overview_df.groupby('age')['amount'].mean().index,overview_df.groupby('age')['amount'].mean())
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Average Transaction ($)')
    ax2.plot(overview_df.groupby('income')['amount'].mean().index,overview_df.groupby('income')['amount'].mean())
    ax2.set_xlabel('Annual Income')
    # fig.suptitle('Average Spend Amount')
    plt.subplots_adjust(hspace = 0.1)
    plt.savefig('avg_spend.png', dpi=300, bbox_inches = "tight")

def avg_spend_count_age_income(df):
    '''
    Plots Average spend for Age and Income
    '''
    fig, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(9,6), sharey = True)
    ax1.plot(overview_df.groupby('age')['amount'].mean().index,overview_df.groupby('age')['amount'].count())
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Transaction Count')
    ax2.plot(overview_df.groupby('income')['amount'].mean().index,overview_df.groupby('income')['amount'].count())
    ax2.set_xlabel('Annual Income')
    # fig.suptitle('Average Spend Amount')
    plt.subplots_adjust(hspace = 0.1)
    plt.savefig('avg_spend_count.png', dpi=300, bbox_inches = "tight")
avg_spend_count_age_income(overview_df)
avg_spend_age_income(overview_df)

overview_df.groupby('offer_index')['amount'].mean()


fig, ax = plt.subplots()
trans = overview_df[overview_df['event_transaction'] == 1].copy()
not_trans = overview_df[overview_df['event_transaction'] == 0].copy()
overview_df['rev'] = overview_df['amount'] - overview_df['reward']
ax.plot(trans.groupby('age')['amount'].mean().index, trans.groupby('age')['amount'].mean().values, label = 'No Offer Transactions')
ax.plot(not_trans.groupby('age')['amount'].mean().index, not_trans.groupby('age')['amount'].mean().values, label = 'Offer Transactions')
ax.plot(overview_df.groupby('age')['rev'].mean().index, overview_df.groupby('age')['rev'].mean().values, label = 'Offer Transactions - Reward')

ax.legend()
