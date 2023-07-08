import pandas as pd


def costume_split_train_test_set(model_case=1):
    data = pd.read_csv('Data/action.csv')
    num_of_patients = len(data['patient'].unique()) + 1
    range_1 = range(1, 24)
    range_2 = range(1, 20)
    range_3 = range(24, 50)
    print(data)
    if model_case == 2:
        num_of_patients = 23
        range_1 = [2, 7, 9, 12, 13, 14, 17, 18]
        range_2 = [2, 7, 9, 12, 13, 18]
        range_3 = [1, 3, 4, 5, 8, 11, 16, 19, 20, 21, 22, 23]

    # we want to spit to train and test set but we want to keep the patients together
    # so we won't have the same patient in both sets
    # but also the train and test sets to have the same number of healthy and depressed patients
    train = pd.DataFrame()
    test = pd.DataFrame()
    for i in range(1, num_of_patients):
        df = data[data['patient'] == i]
        patient = i
        df = df.reset_index(drop=True)
        df = df.drop(['patient'], axis=1)
        header = ['activity', 'target', 'patient_new', 'afftype', 'patient']
        if i in range_1:
            # assign 0.8 of the patients to train set and 0.2 to test set
            if i in range_2:
                df['patient'] = patient
                train = pd.concat([train, df], axis=0)
            else:
                df['patient'] = patient
                test = pd.concat([test, df], axis=0)
        else:
            # assign 0.8 of the patients to train set and 0.2 to test set
            if i in range_3:
                df['patient'] = patient
                train = pd.concat([train, df], axis=0)
            else:
                df['patient'] = patient
                test = pd.concat([test, df], axis=0)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # save the train and test sets to csv
    train.to_csv('Data/train' + str(model_case) + '.csv', index=False)
    test.to_csv('Data/test' + str(model_case) + '.csv', index=False)

    # print(train)
    # print(test)


costume_split_train_test_set(1)
costume_split_train_test_set(2)
