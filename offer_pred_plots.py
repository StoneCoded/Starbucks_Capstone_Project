import pandas as pd
from datetime import datetime
from clean_data import *
from process_data import *
from offer_pred_functions import *
import matplotlib.pyplot as plt


# Get Data
starbucks_df = load_data()
success_df = starbucks_df[starbucks_df['success'] == 1].copy()
fdp_df = pd.read_pickle('./data/full_data_prediction.pkl')
fdp_df['adjstd_amt'] = fdp_df['amount'] * fdp_df['success']
fdp_df['offer_index'] = np.nan
person_df = starbucks_df.groupby(['person_id', 'age','income', 'gender', 'member_length']).count().reset_index().iloc[:,:5].copy()

def gen_graph(file = fdp_df):
    '''
    ARGS    - file (default fdp_df (full_data_prediction.pkl))

    Returns - Bar Graph
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Returns a comparison of total spending between data and prediction
    '''
    f_avg = file.groupby('gender_F')['adjstd_amt'].sum().values[1]
    m_avg = file.groupby('gender_M')['adjstd_amt'].sum().values[1]
    o_avg = file.groupby('gender_O')['adjstd_amt'].sum().values[1]

    over_slice = starbucks_df[['amount','gender']].copy()
    over_slice = pd.get_dummies(over_slice)

    kf_avg = over_slice.groupby('gender_F')['amount'].sum().values[1]
    km_avg = over_slice.groupby('gender_M')['amount'].sum().values[1]
    ko_avg = over_slice.groupby('gender_O')['amount'].sum().values[1]

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
    plt.savefig(f"./plots/output_gender_avg.png", dpi=300, bbox_inches = "tight")
    plt.show()

def overview_graph_basic(df):
    '''
    ARGS: df - DataFrame

    RETURNS: Demographic Graph
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Returns a 2x2 set of histograms showing demographic distribution
    '''

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

    plt.savefig(f"./plots/overview_demo_full.png", dpi=300, bbox_inches = "tight")

def overview_graph(df1, df2, df3, labels, filename):
    '''
    ARGS: df1, df2, df3 - DataFrames
    labels: Name of each data frame for legend

    RETURNS: 2x2 Histrgrams
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Each data frame is split into demographic data and plotted against each other.
    '''

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9,6))
    fig.tight_layout()
    plt.subplots_adjust(hspace = 0.3)

    ax1.hist(df1.age.values, alpha = 0.5)
    ax1.set_title('Age')

    ax2.hist(df1.income.values, alpha = 0.5)
    ax2.set_xticks([40000,60000,80000,100000,120000])
    ax2.set_xticklabels(['40k','60k','80k','100k','120k'])
    ax2.set_title('Annual Income')

    gndr_cnt =[df1[df1.gender == 'M'].shape[0],
               df1[df1.gender == 'F'].shape[0],
               df1[df1.gender == 'O'].shape[0]]

    ax3.bar(['Male','Female','Other'], gndr_cnt, alpha = 0.5)
    ax3.set_title('Gender')

    ax4.hist(df1.member_length.values, alpha = 0.5)
    ax4.set_xticks([0,365,730,1095, 1460, 1825])
    ax4.set_xticklabels(['0','1','2','3','4', '5'])
    ax4.set_title('Membership Length (Years)')

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
    ax3.legend(labels)

    ax4.hist(df3.member_length.values, alpha = 0.5)
    ax4.set_xticks([0,365,730,1095, 1460, 1825])
    ax4.set_xticklabels(['0','1','2','3','4', '5'])
    ax4.set_title('Membership Length (Years)')
    # plt.legend(['Transaction','Successful Offer','Failed Offer'])
    plt.savefig(filename, dpi=300, bbox_inches = "tight")

# #No Offer/Successful Offer/Failed Offer
# overview_graph(starbucks_df[starbucks_df['event_transaction'] == 1],
#                starbucks_df[starbucks_df['event_offer completed'] == 1],
#                starbucks_df[starbucks_df['event_offer received'] == 1],
#                ['No Offer','Successful Offer', 'Failed Offer'],
#                f"event_compare_demo.png")
# #Discount/Bogo/Informational Total
# overview_graph(starbucks_df[starbucks_df['offer_type'] == 'discount'],
#                starbucks_df[starbucks_df['offer_type'] == 'bogo'],
#                starbucks_df[starbucks_df['offer_type'] == 'informational'],
#                ['Discount','BOGO', 'Informational'],
#                f"offer_compare_demo.png")
# #Discount/Bogo/Informational Success
# overview_graph(success_df[success_df['offer_type'] == 'discount'],
#                success_df[success_df['offer_type'] == 'bogo'],
#                success_df[success_df['offer_type'] == 'informational'],
#                ['Discount','BOGO', 'Informational'],
#                f"Successful_offer_compare_demo.png")

def avg_spend_age_income(df):
    '''
    ARGS: df - Data Frame

    RETURNS: 2x1 Line Graphs
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Plots Average spend for Age and Income
    '''
    fig, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(9,6), sharey = True)
    ax1.plot(starbucks_df.groupby('age')['amount'].mean().index,starbucks_df.groupby('age')['amount'].mean())
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Average Transaction ($)')
    ax2.plot(starbucks_df.groupby('income')['amount'].mean().index,starbucks_df.groupby('income')['amount'].mean())
    ax2.set_xlabel('Annual Income')
    # fig.suptitle('Average Spend Amount')
    plt.subplots_adjust(hspace = 0.1)
    plt.show()
    # plt.savefig('avg_spend.png', dpi=300, bbox_inches = "tight")

def avg_spend_count_age_income(df):
    '''
    ARGS: df - Data Frame

    RETURNS: 2x1 Line Graphs
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Plots transaction counts for Age and Income
    '''
    fig, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(9,6), sharey = True)
    ax1.plot(starbucks_df.groupby('age')['amount'].mean().index,starbucks_df.groupby('age')['amount'].count())
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Transaction Count')
    ax2.plot(starbucks_df.groupby('income')['amount'].mean().index,starbucks_df.groupby('income')['amount'].count())
    ax2.set_xlabel('Annual Income')
    plt.subplots_adjust(hspace = 0.1)
    plt.show()
    # plt.savefig('avg_spend_count.png', dpi=300, bbox_inches = "tight")

def fdp_df_sort_difficulty(df = fdp_df):
    '''
    ARGS: None

    RETURNS:None
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Takes the adjstd_amt (amount adjusted) columns and adjusts further depending on
    whether the amount passes the difficulty or not.

    WARNING: As it cycles through each row, it can take upward of 10 minutes to run.
    '''
    offers = starbucks_df.groupby(['offer_index', 'offer_reward', 'difficulty',
                        'duration', 'offer_type', 'offer_id', 'email', 'mobile',
                        'social', 'web']).count().reset_index().iloc[:,:10].copy()
    m = 1
    for n in range(11):
        df.iloc[n::12, 9] = m
        m+=1
    df = df.merge(offers, on = 'offer_index')


    for n in range(len(df)):
        if df.iloc[n, 8] < df.iloc[n, 11]:
            df.iloc[n, 8] = 0
    return df

def success_prediction_plot(df = fdp_df):
    '''
    ARGS: None

    RETURNS: 2x2 Graph Consisting of:
                    - 2 Line Graphs
                    - 1 Histogram
                    - 1 Bar Graph
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Each graph is a comparison between the core data success and the predicted
    '''
    def total_success(p_ax, target):
        '''
        ARGS: p_ax   - Axes Object
              target - Target Demographic

        RETURNS: Line Graph
        –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        Simple function to produce a line graph based on target demographic
        '''
        x1 = starbucks_df.groupby(target)['success'].sum().index
        y1 = starbucks_df.groupby(target)['success'].sum().values

        x2 = df.groupby(target)['success'].sum().index
        y2 = df.groupby(target)['success'].sum().values

        p_ax.plot(x1,y1, label = 'Total Success')
        p_ax.plot(x2,y2, label = 'Predicted Total Success')
        p_ax.set_xlabel(target.capitalize())
        p_ax.set_ylabel('Successes')
        return p_ax

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (10,7))
    ax1 = total_success(ax1, 'age')
    ax2 = total_success(ax2, 'income')
    ax2.legend()

    ax3.hist(fdp_df.groupby('member_length')['success'].sum().values, bins = 20, color = 'orange', alpha = 0.6)
    ax3.hist(starbucks_df.groupby('member_length')['success'].sum().values,bins = 20, color = '#1f77b4', alpha = 0.6)
    ax3.set_ylabel('Successes')
    ax3.set_xlabel('Membership Length (Days)')


    x = ['Female', 'Male', 'Other']
    y = starbucks_df.groupby('gender')['success'].sum().values
    y2 = [df.iloc[:,5].sum(),
    df.iloc[:,6].sum(),
    df.iloc[:,7].sum()]

    ax4.bar(x,y2, color = 'orange')
    ax4.bar(x,y, color = '#1f77b4')
    ax4.set_ylabel('Successes')
    ax4.set_xlabel('Gender')
    plt.subplots_adjust(wspace = 0.3)
    #plt.savefig('tot_suc.png', dpi=300, bbox_inches = "tight")

def amount_prediction_plot(df = fdp_df):
    '''
    ARGS: None

    RETURNS: 2x2 Graph Consisting of:
                    - 2 Line Graphs
                    - 1 Histogram
                    - 1 Bar Graph
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Each graph is a comparison between the core data amount and the predicted
    '''
    def total_amount(p_ax, target):
        '''
        ARGS: p_ax   - Axes Object
              target - Target Demographic

        RETURNS: Line Graph
        –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        Simple function to produce a line graph based on target demographic
        '''
        x1 = starbucks_df.groupby(target)['amount'].sum().index
        y1 = starbucks_df.groupby(target)['amount'].sum().values

        x2 = df.groupby(target)['adjstd_amt'].sum().index
        y2 = df.groupby(target)['adjstd_amt'].sum().values

        p_ax.plot(x1,y1, label = 'Amount')
        p_ax.plot(x2,y2, label = 'Predicted Amount')
        p_ax.set_xlabel(target.capitalize())
        p_ax.set_ylabel('Total Spend($)')
        return p_ax

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (10,7))
    ax1 = total_amount(ax1, 'age')
    ax2 = total_amount(ax2, 'income')
    ax2.legend()

    ax3.hist(df.groupby('member_length')['adjstd_amt'].sum().values,bins = 30, color = 'orange', alpha = 0.6)
    ax3.hist(starbucks_df.groupby('member_length')['amount'].sum().values,bins = 30, color = '#1f77b4', alpha = 0.6)
    ax3.set_ylabel('Total Spend($)')
    ax3.set_xlabel('Membership Length (Days)')

    x = ['Female', 'Male', 'Other']
    y = starbucks_df.groupby('gender')['amount'].sum().values
    y2 = [df[df.iloc[:,5] == 1].adjstd_amt.sum(),
    df[df.iloc[:,6] == 1].adjstd_amt.sum(),
    df[df.iloc[:,7] == 1].adjstd_amt.sum()]

    ax4.bar(x,y2, color = 'orange', alpha = 0.6)
    ax4.bar(x,y, color = '#1f77b4', alpha = 0.6)
    ax4.set_yticks([0,200000,400000,600000,800000,1000000])
    ax4.set_yticklabels(['0','200k','400k','600k','800k','1m'])
    ax4.set_ylabel('Total Spend($)')
    ax4.set_xlabel('Gender')
    plt.subplots_adjust(wspace = 0.3)
    # plt.savefig('tot_am.png', dpi=300, bbox_inches = "tight")
