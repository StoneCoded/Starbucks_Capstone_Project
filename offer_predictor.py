import numpy as np
import pandas as pd
from clean_data import *
from offer_pred_functions import *
from offer_pred_plots import *

class Offer_Predictor():

    '''
    _________________________________________________
    ––_––_––_––_Starbucks Offer Predictor_––_––_––_––
    –––––––––––––––––––––––––––––––—–––––––––––––––––
    Predicts success and transaction amount based off
    of demographic data.

    All sub functions are listed
    below with each having its own doc string.

    Class Functions:
        __init__
        __load__
        __build__
        predict_person
        predict_new_person
        predict_all
        coef_
        recommendation
        rec_plot
        plot_graph
    '''
    def __init__(self):
        '''
        None
        '''

    def __load__(self, clean = False):
        '''
        ARGS: None

        RETURNS:
                 starbucks_df     - full, clean, ready to process DataFrame
                 user_list        - list of all unique person_indexs
        ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        Simple load function.
        '''
        if clean == True:
            save_data()

        self.starbucks_df = load_data('./data/clean_data.pkl')
        self.user_list = self.starbucks_df.person_index.unique().tolist()

    def __build__(self):
        '''
        ARGS:
                df                - DataFrame
        RETURNS:
                success_model     - Dictionary of prediction models for predicting
                                    offer success.
                success_accuracy  - accuracy score of success_model
                success_f1        - f1 score of success_model
                amount_model      - Dictionary of prediction models for predicting
                                    offer amount.
                amount_r2         - r2 score of amount_model
                amount_mse        - mean_squared_error of amount model
        ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        Creates two model dictionaries containing predictions for all offers.

        '''
        self.success_model, self.success_accuracy, self.success_f1, \
        self.amount_model, self.amount_r2, self.amount_mse, self.offers = build_models(self.starbucks_df)

    def predict_person(self, person_idx):
        '''
        ARGS:   person_idx        - Int index of person

        RETURNS:
                predict           - DataFrame of predicted success and amount_s
                                    next to corresponding offer
                offers            - DataFrame of Offer information
                user_value        - Standardised user demographic info for prediction
                user_data         - All rows from full dataframe corresponding to person_idx
                compare           - Combines predict, and offers to show the full
                                    information of each offer next to the corresponding
                                    prediction.
        ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        Main prediction function.
        '''
        if person_idx not in self.user_list:
            print('Sorry friend, this user is not currently in the database')
            print('Try running with new_person, see docstring for help.')
        else:
            self.person_index = person_idx
            self.predictions, self.user_value, self.user_data =  predictor(self.starbucks_df, self.person_index, self.success_model, self.amount_model)
            self.predictions = self.predictions.merge(self.offers, on = 'offer_index')

    def predict_new_person(self, age, income, membership_length, gender):
        '''
        ARGS:
                age               - (int) Age for New Person, must be older than
                                    0 and younger that 120
                income            - (int/float) Income for New Person, must be
                                    between 0 and 140000.0
                membership_length - (int) Membership length in days
                gender            - (str) Gender - 'F' = Female
                                  - 'M' = Male
                                  - 'O' = Other
        RETURNS:
                success_accuracy  - accuracy score of success_model
                success_f1        - f1 score of success_model
                amount_r2         - r2 score of amount_model
                amount_mse        - mean_squared_error of amount model
                predict           - DataFrame of predicted success and amount_s
                                    next to corresponding offer
                offers            - DataFrame of Offer information
                user_value        - Standardised user demographic info for prediction
                user_data         - All rows from full dataframe corresponding to person_idx
                success_model     - Dictionary of prediction models for predicting
                                    offer success.
                amount_model      - Dictionary of prediction models for predicting
                                    offer amount.
                compare           - Combines predict, and offers to show the full
                                    information of each offer next to the corresponding
                                    prediction.
        ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        Simply put, it adds a new person of ARGS design, adds them into the main
        DataFrame then predicts as per usual. To speed this up, the success/amount
        models should've been created already.
        '''

        new_df = pd.DataFrame(columns = ['person_id', 'hours_elapsed', 'amount', 'reward', 'offer_id',
       'days_elapsed', 'person_index', 'offer_index', 'gender', 'age',
       'income', 'offer_reward', 'difficulty', 'duration', 'offer_type',
       'email', 'mobile', 'social', 'web', 'success', 'event_offer completed',
       'event_offer received', 'event_transaction', 'member_length'])

        new_df.loc[999999,:] = 0
        new_df.loc[999999,['person_index', 'age', 'income', 'member_length', 'gender']] = new_person_check(999999, age, income, membership_length, gender)

        self.person_index = 999999
        self.extra_df = self.starbucks_df.copy()
        self.extra_df.loc[999999] = new_df.xs(999999)

        self.predictions, self.user_value, self.user_data =  predictor(self.extra_df, 999999, self.success_model, self.amount_model)
        self.compare = self.predictions.merge(self.offers, on = 'offer_index')

    def predict_all(self, filename = './data/full_data_prediction.pkl'):

        '''
        ####################-TIME CONSUMPTION WARNING-##########################
                             (it takes its sweet time)
        ARGS: None

        RETURNS: None
        ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        Predicts entire DataFrame and saves demographic data to
        './data/amount_prediction_full.pkl'. This process takes a while, at every
        1000 files a print statement lets you know how many thousand are left to
        predict (On my laptop, 2-3 hours).
        '''
        check = input("Are you sure you want to continue? It takes a while, enter Y to continue")
        if check != 'Y':
            print('function abandoned...')

        else:

            ages = []
            amounts = []
            success = []
            income = []
            member_length = []

            gender_F = []
            gender_M = []
            gender_0 = []

            end = (len(self.user_list))

            for n in self.user_list:
                self.predict_new, self.user_value = predict_all(pre_pred_data(self.starbucks_df), \
                            offer_dict = self.success_model, amount_dict = self.amount_model, \
                            person_idx = n, offer_idx_list = self.offers.offer_index.tolist())
                amounts.extend(self.predict_new.iloc[:,2])
                success.extend(self.predict_new.iloc[:,1])

                for x in range(11):
                    ages.append(self.starbucks_df[self.starbucks_df.person_index == n].loc[:,'age'].max())
                    income.append(self.starbucks_df[self.starbucks_df.person_index == n].loc[:,'income'].max())
                    member_length.append(self.starbucks_df[self.starbucks_df.person_index == n].loc[:,'member_length'].max())
                    gen = self.starbucks_df[self.starbucks_df.person_index == n].loc[:,'gender'].max()

                    if gen == 'F':
                        gender_F.append(1)
                        gender_M.append(0)
                        gender_0.append(0)
                    elif gen == 'M':
                        gender_F.append(0)
                        gender_M.append(1)
                        gender_0.append(0)
                    elif gen == 'O':
                        gender_F.append(0)
                        gender_M.append(0)
                        gender_0.append(1)
                end -= 1
                if (end % 1000 == 0):
                    print(end)
            d = {'amount':amounts, 'success':success, 'age':ages, 'income':income, 'member_length':member_length, 'gender_F':gender_F, 'gender_M':gender_M, 'gender_O':gender_0}
            amount_prediction = pd.DataFrame(d)
            amount_prediction.to_pickle(filename)
        print('finished.')

    def coef_(self):
        '''
        ARGS: None

        RETURNS: success_coef - coef_ of LogisticRegression model
                 amount_coef  - coef_ of Ridge regressor model
        –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        Shows features weights for each model for each offer.

        '''

        self.success_coef = pd.DataFrame(columns = self.user_value.columns)
        for n in self.success_model:
            self.success_coef = self.success_coef.append(pd.DataFrame(self.success_model[n].coef_, columns = self.user_value.columns))
        self.success_coef.index = range(1, len(self.success_model) + 1)

        am_coef = []
        for n in (range(1, len(self.amount_model) + 1)):
            am_coef.append(self.amount_model[f'offer_{n}'].coef_)
        self.amount_coef  = pd.DataFrame(am_coef, columns = self.user_value.columns)

    def recommendation(self):
        '''
        ARGS: None

        RETURNS:
        –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        Take predictions and offer parameters into account and provides offer recommendation
        for predicted person.

        '''
        recs = self.predictions.copy()

        def f(x):
          return 1 if x['predicted_amount'] > x['difficulty'] else 0

        recs['diff_check'] = recs.apply(f, axis=1)
        recs['success_adjust'] = recs['predicted_success'] * recs['diff_check']
        recs['amount_adjust'] = recs['predicted_amount'] * recs['success_adjust']
        recs = recs.sort_values(by='amount_adjust', ascending = False).reset_index(drop = True)
        self.recommend = recs[recs['success_adjust'] == 1].iloc[:,:-3]
        return self.recommend

    def rec_plot(self):
        fig, ax = plt.subplots()
        self.user_data.person_index.values[0]
        x = self.starbucks_df[self.starbucks_df['person_index'] == self.person_index].amount.index
        y = self.starbucks_df[self.starbucks_df['person_index'] == self.person_index].amount.values

        x2 = self.recommend.predicted_amount.index
        y2 = self.recommend.predicted_amount.values

        ax.scatter(range(len(x)), y, label = 'Transactions')
        ax.scatter(x2, y2, label = 'Predicted Transactions')
        ax.set_ylabel('Amount($)')
        ax.set_xlabel('Transaction Number')
        ax.legend()
        plt.show()

    def plot_graph(self, gender = False, over_basic = False, event_comp = False,
                    offer_comp = False, succ_comp = False, avg_spend = False,
                    avg_spend_count = False, success_prediction = False,
                    amount_prediction = False):
        '''
        ARGS/RETURNS:
            gender             - A comparison of total spending between data and
                                 prediction.
            over_basic         - 2x2 Set of histograms showing demographic
                                 distribution.
            event_comp         - Comparison between No, Successful and Failed Offers.
            offer_comp         - Shows distribution of each type of Offer.
            succ_comp          - Shows distribution of each type of Successful Offer.
            avg_spend          - Shows average spending by Age and Income.
            avg_spend_count    - Shows transaction amount by Age and Income.
            success_prediction - Overall Comparison between success values and
                                 predicted values by demographic.
            amount_prediction  - Overall Comparison between transaction values
                                 and predicted values by demographic.
        –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        All ARGS default = False.

        Note: If predicted amount is less than difficulty then offer will fail.
              This has not been applied to the full comparison data. It has been
              applied for the results (see REAME).
        '''
        if gender == True:
            gen_graph()
        if over_basic == True:
            overview_graph_basic(self.starbucks_df)
        if event_comp == True:
            overview_graph(starbucks_df[starbucks_df['event_transaction'] == 1],
                           starbucks_df[starbucks_df['event_offer completed'] == 1],
                           starbucks_df[starbucks_df['event_offer received'] == 1],
                           ['No Offer','Successful Offer', 'Failed Offer'],
                           f"./plots/event_compare_demo.png")
        if offer_comp == True:
            overview_graph(starbucks_df[starbucks_df['offer_type'] == 'discount'],
                           starbucks_df[starbucks_df['offer_type'] == 'bogo'],
                           starbucks_df[starbucks_df['offer_type'] == 'informational'],
                           ['Discount','BOGO', 'Informational'],
                           f"./plots/offer_compare_demo.png")
        if succ_comp == True:
            overview_graph(success_df[success_df['offer_type'] == 'discount'],
                           success_df[success_df['offer_type'] == 'bogo'],
                           success_df[success_df['offer_type'] == 'informational'],
                           ['Discount','BOGO', 'Informational'],
                           f"./plots/Successful_offer_compare_demo.png")
        if avg_spend == True:
            avg_spend_age_income(self.starbucks_df)
        if avg_spend_count == True:
            avg_spend_count_age_income(self.starbucks_df)
        # if apply_difficulty == True:
        #     fdp_df = fdp_df_sort_difficulty()
        # Not Working Right Now
        if success_prediction == True:
            success_prediction_plot()
        if amount_prediction == True:
            amount_prediction_plot()
