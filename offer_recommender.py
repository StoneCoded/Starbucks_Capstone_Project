import numpy as np
import pandas as pd
from offer_rec_functions import *

import progressbar


class Offer_Recommender():

    '''
    Intended Result: Recommend Best Offer For Money
    '''
    def __init__(self):
        '''
        None
        '''
    def __load__(self):
        '''
        loads data
        '''
        self.starbucks_df = load_data('./data/clean_data.pkl')

    def create_new_person(self, index, age, income, member_length, gender):
        '''

        '''
        new_df = pd.DataFrame(columns = ['person_id', 'hours_elapsed', 'amount', 'reward', 'offer_id',
               'days_elapsed', 'person_index', 'offer_index', 'gender', 'age',
               'income', 'offer_reward', 'difficulty', 'duration', 'offer_type',
               'email', 'mobile', 'social', 'web', 'success', 'event_offer completed',
               'event_offer received', 'event_transaction', 'member_length'])

        new_df.loc[999999,:] = 0
        new_df.loc[999999,['person_index', 'age', 'income', 'member_length', 'gender']] = [index, age, income, member_length, gender]
        self.extra_df = self.starbucks_df.append(new_df.xs(999999))

        return self.extra_df

    def offer_predict(self, person_idx, new_person = False):
        '''
        All Predictions
        '''
        if new_person == True:
            full_df = self.extra_df.copy()
        else:
            full_df = self.starbucks_df.copy()
        self.success_accuracy, self.amount_r2, self.predict, self.offers, \
        self.user_value, self.user_data, self.success_model, self.amount_model\
                 =  predictor(self.starbucks_df, full_df, person_idx)

    def coef_(self):
        '''
        Gets All coef_

        '''

        self.success_coef = pd.DataFrame(columns = self.user_value.columns)
        for n in self.success_model:
            self.success_coef = self.success_coef.append(pd.DataFrame(self.success_model[n].coef_, columns = self.user_value.columns))
        self.success_coef.index = range(1, len(self.success_model) + 1)

        am_coef = []
        for n in (range(1, len(self.amount_model) + 1)):
            am_coef.append(self.amount_model[f'offer_{n}'].coef_)
        self.amount_coef  = pd.DataFrame(am_coef, columns = self.user_value.columns)

#
# t = Offer_Recommender()
# t.__load__()
# t.offer_predict(1)
# d = {'offer_index' : range(1,12)}
# t.starbucks_df.describe()
# t.offers
# t.create_new_person(999999, 28, 30000, 20, 'F')
# t.offer_predict(999999, new_person = True)
# t.coef_()
# t.success_coef
# success_test = pd.DataFrame(d)
# amount_test = pd.DataFrame(d)
# # for n in range(0, 10000, 50):
# for n in ['M', 'F', 'O']:
#     t.create_new_person(999999, 53, 62000, 624, n)
#     t.offer_predict(999999, new_person = True)
#
#     success_test = success_test.merge(t.predict.iloc[:,[0,1]], on = 'offer_index')
#     amount_test = amount_test.merge(t.predict.iloc[:,[0,2]], on = 'offer_index')
#
# success_test = success_test.set_index('offer_index')
# # success_test.columns = list(range(0, 10000, 50))
# success_test.columns = ['M', 'F', 'O']
#
# amount_test = amount_test.set_index('offer_index')
# # amount_test.columns = list(range(0, 10000, 50))
# amount_test.columns = ['M', 'F', 'O']
#
# amount_test.to_pickle('amount_gender_test.pickle')
# success_test.to_pickle('success_gender_test.pickle')




'''

The basic task is to use the data to identify which groups of people are
most responsive to each type of offer, and how best to present each type of offer

'''
