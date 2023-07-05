# import libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


data = pd.DataFrame()

df_scores = pd.read_csv('Data/scores.csv')

df_scores['age'] = df_scores['age'].replace(["20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                                                "55-59", "60-64","65-69"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df_scores['edu'] = df_scores['edu'].replace(["6-10", "11-15", "16-20"], [1, 2, 3])

df_scores = df_scores.drop(['madrs1', 'madrs2', 'days', 'afftype', 'melanch', 'inpatient', 'edu', \
                            'marriage', 'work', 'number'], axis=1)


# create a target column
df_scores['target'] = 0

# set the target to 1 if the patient is healthy and 0 if the patient is depressed
for i in range(1, 24):
    df_scores['target'][i] = 0

for i in range(1, 33):
    df_scores['target'][i+23] = 1

# save scores to csv 
df_scores.to_csv('Data/scores_for classification_1.csv', index=False)

# healthy = 1, depressed = 0
for i in range(1, 24):
    file = 'data/condition/condition_' + str(i) + '.csv'
    df = pd.read_csv(file)
    df['patient'] = i
    df['target'] = 0


    data = pd.concat([data, df], axis=0)


for i in range(1, 33):
    file = 'Data/control/control_' + str(i) + '.csv'
    df = pd.read_csv(file)
    df['target'] = 1  
    df['patient'] = i+23 

    
    data = pd.concat([data, df], axis=0)


data = data.drop(['timestamp', 'date'], axis=1)

cols = list(data.columns)
data = data[[cols[1]]+[cols[0]]+[cols[2]]]

# separate the features that belong to the same patient
data_for_feat_extr = data.groupby('patient').agg(list)

# change the [1,1,1,...] to 1 and [0,0,0,...] to 0
for i in range(1, 56):
    data_for_feat_extr['target'][i] = data_for_feat_extr['target'][i][0]

# save data to csv
data_for_feat_extr.to_csv('Data/data_for_feat_extr.csv', index=False)