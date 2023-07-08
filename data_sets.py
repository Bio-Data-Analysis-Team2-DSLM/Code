import pandas as pd

data = pd.read_csv('Data/action.csv')

print(data)

# we want to spit to train and test set but we want to keep the patients together
# so we won't have the same patient in both sets
# but also the train and test sets to have the same number of healthy and depressed patients
train = pd.DataFrame()
test = pd.DataFrame()
for i in range(1, len(data['patient'].unique()) + 1):
    df = data[data['patient'] == i]
    patient = i
    df = df.reset_index(drop=True)
    df = df.drop(['patient'], axis=1)
    header = ['activity', 'target', 'patient_new', 'afftype', 'patient']
    if i in range(1, 24):
        # assign 0.8 of the patients to train set and 0.2 to test set
        if i in range(1, 20):
            df['patient'] = patient
            train = pd.concat([train, df], axis=0)
        else:
            df['patient'] = patient
            test = pd.concat([test, df], axis=0)
    else:
        # assign 0.8 of the patients to train set and 0.2 to test set
        if i in range(24, 50):
            df['patient'] = patient
            train = pd.concat([train, df], axis=0)
        else:
            df['patient'] = patient
            test = pd.concat([test, df], axis=0)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# save the train and test sets to csv
train.to_csv('Data/train.csv', index=False)
test.to_csv('Data/test.csv', index=False)

print(train)
print(test)