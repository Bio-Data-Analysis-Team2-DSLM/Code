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

data = data.drop(['date'], axis=1)
#print(data)

data['timestamp'] = data['timestamp'].str[11:]
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%H:%M:%S').dt.time


#for i in range(1, 56):
   # print(data[data['patient'] == i])


# we remove the rows before the first 00:00:00 and after the last 00:00:00 for each patient
data2 = pd.DataFrame()
for i in range(1, 56):
    #print(data[data['patient'] == i])
    df = data[data['patient'] == i]
    patient = i
    df = df.reset_index(drop=True)

    for j in range(0, len(df)):
        if df['timestamp'][j] == pd.to_datetime('00:00:00', format='%H:%M:%S').time():
            first = j
            break

    for j in range(len(df)-1, -1, -1):
        if df['timestamp'][j] == pd.to_datetime('23:59:00', format='%H:%M:%S').time():
            last = j
            break
    df['patient'] = patient
    df = df[first:last+1]
    #df = df.reset_index(drop=True)
    #data = data.drop(data[data['patient'] == i].index)
    data2 = pd.concat([data2, df], axis=0)

data2 = data2.reset_index(drop=True)
data2.to_csv('Data/action.csv', index=False)