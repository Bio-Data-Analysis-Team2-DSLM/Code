# we will simplify the data by taking the mean of every 30 values of the activity
# for every patient
# we will drop the rows that are not divisible by 30
import pandas as pd

data = pd.read_csv('Data/action.csv')

for i in range(1, 56):
    df = data[data['patient'] == i]
    df = df.reset_index(drop=True)
    df = df.drop(['patient'], axis=1)
    df = df.drop(['timestamp'], axis=1)
    df = df.drop(['target'], axis=1)
    for j in range(0, len(df), 30):
        df.loc[j] = df.loc[j:j+30].mean()
    df['patient'] = i

    data = pd.concat([data, df], axis=0)

data = data.reset_index(drop=True)
print(data)
