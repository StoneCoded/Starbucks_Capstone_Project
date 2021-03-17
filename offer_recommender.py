import numpy as np
import pandas as pd
from offer_rec_functions import *

class Offer_Recommender:

    '''
    Intended Result: Recommend Best Offer For Money
    '''
    def __init__(self):
        '''
        None

        '''
        self.filename = './data/dropped_clean_data.pkl'
        self.starbucks_df = load_data(self.filename)
        self.user_df = pre_pred_data(self.starbucks_df)

    def offers(self):
        portfolio = self.starbucks_df[['offer_reward', 'difficulty', 'duration',
                            'offer_type', 'offer_id', 'email', 'mobile', 'social',
                            'web', 'offer_index']]

        self.offers = portfolio.groupby(['offer_index', 'offer_reward', 'difficulty',
                            'duration', 'offer_type', 'offer_id', 'email', 'mobile',
                            'social', 'web']).count().reset_index()
        return self.offers


    def predict(self, user_index):
        try:
            self.trans_id = self.offers[self.offers['offer_type'] == 'transaction'].iloc[0,0]
            self.offer_success_model, self.success_accuracy = predict_offers(self.starbucks_df, self.trans_id)
            self.offer_amount_model, self.amount_r2 = predict_amount(self.starbucks_df, self.trans_id)
            self.user_index = user_index
            self.predictions = predict(self.user_index, df = self.user_df, offer_dict = self.offer_success_model, amount_dict = self.offer_amount_model)
            self.predictions = self.predictions.sort_values(by = 'predicted_amount', ascending = False).reset_index(drop = True)

            return self.predictions
        except:
            print("I'm sorry, but a prediction cannot be made for this person.")
            print("It looks like this person does not exist in the current database.")

            return None

t = Offer_Recommender()
t.offers()
t.predict(100)
