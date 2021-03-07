import pandas as pd
from datetime import datetime
from clean_data import *
from process_data import *
import matplotlib.pyplot as plt

from scipy.stats.stats import pearsonr
pd.set_option('mode.chained_assignment', 'raise')


transcript_df, portfolio_df, profile_df = clean_data()

part_over_df = transcript_df.merge(profile_df, on= ['person_id', 'person_index'], how = 'outer')
overview_df = part_over_df.merge(portfolio_df, on= ['offer_id', 'offer_index'], how = 'outer')

overview_df = process_df(overview_df)
