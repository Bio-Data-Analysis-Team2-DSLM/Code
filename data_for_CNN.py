import pandas as pd
import numpy as np

data = pd.read_csv('Data/action.csv')

#print(data.head())

# we will simplify the data by taking the mean of every 30 values of the activity
# for every patient
# we will drop the rows that are not divisible by 30

data2 = pd.DataFrame()

for i in range(1, 56):
    df = data[data['patient'] == i]
    df = df.reset_index(drop=True)
    #print(df)
    #print(len(df))
    if len(df) % 30 != 0:
        df = df.drop(df.index[len(df) - len(df) % 30:])
    #print(len(df))
    df = df.groupby(np.arange(len(df))//30).mean(numeric_only=True)
    df = df.round(3)
    df['patient'] = i
    #print(df)
    data2 = pd.concat([data2, df], axis=0)

data2 = data2.reset_index(drop=True)
#print(data2)

# one day has 1440 minutes and we have one measurement every 30 minutes
# so we have 48 measurements per day
# for every patient wi will keep the maximum amount of days that is divisible by 48

data3 = pd.DataFrame()

for i in range(1, 56):
    df = data2[data2['patient'] == i]
    df = df.reset_index(drop=True)
    #print(df)
    #print(len(df))
    if len(df) % 48 != 0:
        df = df.drop(df.index[len(df) - len(df) % 48:])
    #print(len(df))
    df['patient'] = i
    #print(df)
    data3 = pd.concat([data3, df], axis=0)

data3 = data3.reset_index(drop=True)
print(data3)
data3.to_csv('Data/action_cnn.csv', index=False)