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


data3 = pd.DataFrame()

for i in range(1, 56):
    df = data2[data2['patient'] == i]
    df = df.reset_index(drop=True)
    #print(df)
    #print(len(df))
    #print(len(df))
    df = df.groupby(np.arange(len(df))//30).mean(numeric_only=True)
    df = df.round(3)
    df['patient'] = i
    #print(df)
    data3 = pd.concat([data3, df], axis=0)

data3 = data3.reset_index(drop=True)
print(data3)

# one day has 1440 minutes and we have one measurement every 30 minutes
# so we have 48 measurements per day
# for every patient wi will keep the maximum amount of days that is divisible by 48

##################################################################
##################################################################

# find the index for the first and the last day for every patient



last_day = []

for i in range(1, 56):
    df = data2[data2['patient'] == i]
    df = df.reset_index(drop=True)
    last_day.append(df.index[-1])


first_day = [0]

for i in range(1, 55):
    first_day.append(first_day[i-1] + last_day[i-1] + 1)

print(first_day)
header = ['first_day']

# save the first day to a csv file
df = pd.DataFrame(first_day, columns=header)
df.to_csv('Data/first_row_for_each_patient.csv', index=False)
first_row = pd.read_csv('Data/first_row_for_each_patient.csv')

##################################################################
##################################################################

for i in range(0, len(data), 48):
    data3.loc[i:i+48, 'patient_new'] = i/48 + 1

print(data3)
# save the data to a csv file
data3.to_csv('Data/action.csv', index=False)








