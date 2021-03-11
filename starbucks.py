import pandas as pd
from datetime import datetime
from clean_data import *
from process_data import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from scipy.stats.stats import pearsonr
import pickle

full_df = clean_data()

full_df

a = full_df.pred_income.cumsum().to_frame()
b = full_df.income.cumsum().to_frame()
for frame in [a, b]:
    plot = plt.plot(frame.index.values, frame.iloc[:,0].values, alpha = 0.6)
    plt.xlabel("Income")
    plt.ylabel("")
    plt.title("Income vs Predicted Income")
    plt.legend(('Income', 'Predicted Income'))

plt.savefig(f"output{'_income'}.png", dpi=300, bbox_inches = "tight")
plt.show()
