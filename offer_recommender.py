import numpy as np
import pandas as pd
from offer_rec_functions import *

import progressbar


class Offer_Recommender():

    '''
    Recomendation Engine for Starbucks Offers.
    '''
    def __init__(self):
        '''
        None
        '''
    def __load__(self):
        '''
        ARGS: None

        RETURNS:
                 starbucks_df     - full, clean, ready to process DataFrame
                 user_list        - list of all unique person_indexs
        ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        Simple load function.
        '''

        self.starbucks_df = load_data('./data/clean_data.pkl')
        # self.starbucks_df = pd.read_pickle('./data/test.pkl')
        self.user_list = t.starbucks_df.person_index.unique().tolist()

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

        self.extra_df = self.starbucks_df.copy()
        self.extra_df.loc[999999] = new_df.xs(999999)

        self.predictions, self.user_value, self.user_data =  predictor(self.extra_df, 999999, self.success_model, self.amount_model)
        self.compare = self.predictions.merge(self.offers, on = 'offer_index')

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
            self.predictions, self.user_value, self.user_data =  predictor(self.starbucks_df, person_idx, self.success_model, self.amount_model)
            self.compare = self.predictions.merge(self.offers, on = 'offer_index')

    def person_predict_all(self, filename = './data/amount_prediction_full_3.pkl'):

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

            end = (len(t.user_list))

            for n in t.user_list:
                self.predict_new, self.user_value = predict_all(pre_pred_data(self.starbucks_df), \
                            offer_dict = self.success_model, amount_dict = self.amount_model, \
                            person_idx = n, offer_idx_list = self.offers.offer_index.tolist())
                amounts.extend(t.predict_new.iloc[:,2])
                success.extend(t.predict_new.iloc[:,1])

                for x in range(11):
                    ages.append(t.starbucks_df[t.starbucks_df.person_index == n].loc[:,'age'].max())
                    income.append(t.starbucks_df[t.starbucks_df.person_index == n].loc[:,'income'].max())
                    member_length.append(t.starbucks_df[t.starbucks_df.person_index == n].loc[:,'member_length'].max())
                    gen = t.starbucks_df[t.starbucks_df.person_index == n].loc[:,'gender'].max()

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

t = Offer_Recommender()
t.__load__()
t.__build__()
t.offer_predict(100)
t.compare
t.create_new_person(580, 1800000, 60, 'O')
t.predictions

'''
____________________________Tomorrow's Plan_____________________________________
Final Comparison.
    - Whilst running, recreate new person predict
    - Watch heroku deployment vid
    - Write skeleton report (find out what)
________________________________________________________________________________
'''
