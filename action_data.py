# we will merge the condition and control dataframes
# and we will create the action.csv file that will contain the features
# for every patient
# we will keep the features which correspond to the first timestamp with 
# time 00:00:00 and we will remove the previous ones
# The dataframe will end at the last timestamp with time 00:00:00 - excluding this last 00:00:00

import pandas as pd
import numpy as np

data = pd.DataFrame()
# date has format: 2019-03-25 00:00:00
# we will keep only the time

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

data = data.drop([ 'date'], axis=1)

data['timestamp'] = data['timestamp'].str[11:]
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%H:%M:%S').dt.time


# we remove the rows before the first 00:00:00 and after the last 00:00:00 for each patient
for i in range(1, 56):
    df = data[data['patient'] == i]
    df = df.reset_index(drop=True)
    for j in range(len(df)):
        if df['timestamp'][j] == pd.to_datetime('00:00:00', format='%H:%M:%S').time():
            first_index = j
            break
    for j in range(len(df)):
        if df['timestamp'][j] == pd.to_datetime('00:00:00', format='%H:%M:%S').time():
            last_index = j
            break

    df = df[first_index:last_index]
    df = df.reset_index(drop=True)
    data = pd.concat([data, df], axis=0)

data = data.reset_index(drop=True)
data.to_csv('Data/action.csv')

print(data)