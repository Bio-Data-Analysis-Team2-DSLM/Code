# import libraries
import pandas as pd

data = pd.DataFrame()

df_scores = pd.read_csv('Data/scores.csv')

df_scores['age'] = df_scores['age'].replace(["20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                                                "55-59", "60-64","65-69"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df_scores['edu'] = df_scores['edu'].replace(["6-10", "11-15", "16-20"], [1, 2, 3])


df_scores1 = df_scores.drop(['madrs1', 'madrs2', 'days', 'afftype', 'melanch', 'inpatient', 'edu', \
                            'marriage', 'work', 'number'], axis=1)


# create a target column
df_scores1['target'] = 0

# set the target to 1 if the patient is healthy and 0 if the patient is depressed
for i in range(1, 24):
    df_scores1['target'][i] = 0

for i in range(1, 33):
    df_scores1['target'][i+23] = 1

# save scores to csv 
df_scores1.to_csv('Data/scores_for classification_1.csv', index=False)

df_scores2 = df_scores.drop(['madrs1', 'madrs2', 'days', 'number', 'edu', 'marriage', \
                             'work'], axis=1)

# there ar no enough patients with melancholic depression
# so we will remove this column as well
df_scores2 = df_scores2.drop(['melanch'], axis=1)

df_scores2['target'] = 0

for i in range(1, 24):
    df_scores2['target'][i] = 0

for i in range(25, 55):
    # remove the patients
    df_scores2 = df_scores2.drop([i], axis=0)

print(df_scores2)

# patients 23 and 24 have missing values
df_scores2 = df_scores2.drop([23, 24], axis=0)

print(df_scores2)
# save scores to csv
df_scores2.to_csv('Data/scores_for classification_2.csv', index=False)

# healthy = 1, depressed = 0
for i in range(1, 24):
    file = 'Data/condition/condition_' + str(i) + '.csv'
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