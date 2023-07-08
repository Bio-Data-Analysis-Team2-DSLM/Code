# We will create a simple NN which will take the extracted features from the 1D CNN
# merged with the scores of the patients and will classify them as healthy or depressed.

# import libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import balance_dataset


analytic_train = pd.read_csv('Data/train1.csv')
analytic_test = pd.read_csv('Data/test1.csv')

analytic_train = analytic_train.drop(['afftype'], axis=1)
analytic_test = analytic_test.drop(['afftype'], axis=1)

train_features = pd.read_csv('Data/extracted_features_for_train1.csv')
test_features = pd.read_csv('Data/extracted_features_for_test1.csv')


# function to find the first patient with target 1 (healthy)
# and assign
def assign_targets(analytic, extracted):
    for i in range(0, len(analytic)):
        if analytic['target'][i] == 1:
            first_healthy = i
            break
    # assign the target column
    extracted['target'] = 0
    for i in range(0, int(first_healthy/48)):
        extracted.loc[i, 'target'] = 0 
    for i in range(int(first_healthy/48), len(extracted)):
        extracted.loc[i, 'target'] = 1
    print(extracted)
    return extracted


print('For train set:')
df_train = assign_targets(analytic_train, train_features)
print('For test set:')
df_test = assign_targets(analytic_test, test_features)

# balance the dataset
df_train = balance_dataset.balance_dataset(df_train)
df_test = balance_dataset.balance_dataset(df_test)


# scale the data from all the features except the target
scaler = StandardScaler()
df_train.iloc[:, 1:-1] = scaler.fit_transform(df_train.iloc[:, 1:-1])
df_test.iloc[:, 1:-1] = scaler.transform(df_test.iloc[:, 1:-1])

# split to features and targets
X_train = df_train.drop(['target'], axis=1)
y_train = df_train['target']
X_test = df_test.drop(['target'], axis=1)
y_test = df_test['target']

#------------------------------------------------------------------#
#------------------------------------------------------------------#
#------------------------------------------------------------------#
#------------------------------------------------------------------#
#------------------------------------------------------------------#


# create a Neural Network that will take the features and the scores and will classify them
# to healthy or depressed patients 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # we have classification problem so we will use sigmoid function
        x = torch.sigmoid(self.fc3(x))
        return x

# set the seeds
torch.manual_seed(42)
np.random.seed(42)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print()
print(f'We have {len(X_train)} patients in the trainig set')
print(f'and {len(X_test)} patients in the test set')
print()
print('-----------------------------')

# convert the data to tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)



# create the loss function and the optimizer. We will use MSE loss function
criterion = nn.BCELoss() # Binary Cross Entropy loss for binary classification

# run model_1.py for different hyperparameters 
# and save the results in a csv file
# lr = learning rate
# wd = weight decay
# mm = momentum
# ld = learning rate decay



                
# csv with colums: epochs, lr, weight_decay, momentum, accuracy
# in order to find the best hyperparameters
df = pd.read_csv('outputs/model_1/NN_hyperparameters.csv')



df = pd.concat([df, pd.DataFrame([[epochs, hyperparameters['lr'], hyperparameters['weight_decay'], \
                                                    hyperparameters['momentum'], lr_decay, accuracy]], columns=['epochs', 'lr', \
                                                    'weight_decay', 'momentum', 'lr_decay',  'accuracy'])], axis=0, ignore_index=True)

# change the order of the columns
df = df[['epochs', 'lr', 'weight_decay', 'momentum', 'lr_decay',  'accuracy']]

df.to_csv('outputs/model_1/NN_hyperparameters.csv', index=False, header=True)

# save the model's weights in order to plot the features with their weights
torch.save(net.state_dict(), 'outputs/model_1/model_1_weights.pth')







