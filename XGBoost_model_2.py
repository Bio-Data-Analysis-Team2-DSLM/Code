from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from balance_dataset import balance_dataset

dataset = pd.read_csv("Data/handcraft_data.csv")
dataset = dataset[dataset["patient"] < 24]

tabular_data = pd.read_csv("Data/scores_for classification_2.csv")

age = []
gender = []
target = []
inpatient = []
for i in range(0, len(dataset)):
    real_patient = dataset.loc[i]["patient"]
    age.append(tabular_data.iloc[int(real_patient)-1]["age"])
    gender.append(tabular_data.iloc[int(real_patient) - 1]["gender"])
    t = tabular_data.iloc[int(real_patient)-1]["afftype"]
    if int(t) == 1 or int(t) == 3:
        target.append(0)
    else:
        target.append(1)
    inpatient.append(tabular_data.iloc[int(real_patient)-1]["inpatient"])

dataset["age"] = age
dataset["gender"] = gender
dataset["target"] = target
dataset["inpatient"] = inpatient


dataset = balance_dataset(dataset)


X = dataset.loc[:, ~dataset.columns.isin(['patient', 'target'])]
Y = dataset['target']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Grid search
# params = {
#     'min_child_weight': [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 15],
#     'gamma': [0, 0.3, 0.5, 1, 1.3, 1.4, 1.5, 1.6, 1.7, 2, 5],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
#     'max_depth': [1, 2, 3, 4, 5, 6],
#     'learning_rate': [0.02, 0.01, 0.05, 0.1, 0.4, 0.5, 0.6]
# }
#
# param_comb = 10
# kfold = KFold(n_splits=10, shuffle=True, random_state=2023)
# xgb = xgboost.XGBClassifier(n_estimators=600, objective='binary:logistic',
#                             nthread=1)
# random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy', n_jobs=4,
#                                    cv=kfold.split(X, Y), verbose=3, random_state=2023)
#
# random_search.fit(X, Y)
# print(random_search.best_score_)
# print('\n Best hyperparameters:')
# print(random_search.best_params_)

model = xgboost.XGBClassifier(learning_rate=0.05, n_estimators=600, objective='binary:logistic',
                              nthread=1, max_depth=2, min_child_weight=2, gamma=0.3, colsample_bytree=1.0,
                              subsample=1.0, random_state=2023)
kfold = KFold(n_splits=10, shuffle=True, random_state=2022)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%%" % (results.mean() * 100))
