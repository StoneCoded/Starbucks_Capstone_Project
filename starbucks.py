import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from scipy.stats.stats import pearsonr

import pickle
ordinal_today = pd.to_datetime(datetime.today().strftime('%m/%d/%Y')).toordinal()
reg_today = pd.to_datetime(datetime.today().strftime('%m/%d/%Y'))

starbucks_df = pd.read_pickle('./clean_data.pkl')

starbucks_df = starbucks_df.drop(['pred_age', 'pred_income', 'pred_gender'], axis = 1)
starbucks_df['event_offer completed'] = np.where((starbucks_df['event_offer received'] == 0) & (starbucks_df['event_transaction'] == 0), 1, 0)

starbucks_df['member_length'] = ordinal_today - starbucks_df['ordinal_ms']
starbucks_df = starbucks_df.drop(['ordinal_ms'], axis = 1)

starbucks_df['offer_type'].value_counts()




'''
For next time.

Get prediction models for each offer type
    - potentially groupby each person, summing info to save space
    - dont include person_index

Build person class that can be predicted upon.
We want to show the chance of success by each offer type
    - extending this if we can predict amount spent
    - Also maybe best chance of succes by time sent out

'''
# forest_play
bogo = starbucks_df[starbucks_df['offer_type']=='bogo'].copy()

X_data = starbucks_df[['gender', 'offer_reward', 'difficulty', 'duration', 'member_length','age_bracket', 'income_bracket']]
X_data = pd.get_dummies(X_data, columns = ['gender', 'age_bracket', 'income_bracket'], drop_first = True)

y = starbucks_df['success']

X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier(random_state = 42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
model.score(X_test, y_test)
