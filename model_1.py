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
    return extracted


df_train = assign_targets(analytic_train, train_features)
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
        self.fc1 = nn.Linear(30, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # we have classification problem so we will use sigmoid function
        x = torch.sigmoid(self.fc3(x))
        return x


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


##########################################################################################
##########################################################################################
##########################################################################################


def train_test(X_train, X_test, y_train, y_test, epochs, hyperparameters, lr_decay):
    lr = hyperparameters['lr']
    # set the seeds
    torch.manual_seed(42)
    np.random.seed(42)
    # create the model
    net = Net()
    # create the optimizer
    optimizer = torch.optim.SGD(net.parameters(), **hyperparameters)
    # create the loss function
    criterion = nn.BCELoss() # Binary Cross Entropy loss for binary classification
    
    # create the lists for the loss and accuracy
    train_losses = []
    test_losses = []
    test_accuracy = []
    
    # train the model
    for epoch in range(epochs):
        epoch += 1
        # set the model to train mode
        net.train()
        # clear the gradients
        optimizer.zero_grad()
        # make the predictions
        y_pred = net(X_train)
        # calculate the loss
        loss = criterion(y_pred, y_train.unsqueeze(1).float())
        # backpropagation
        loss.backward()
        # update the weights
        optimizer.step()
        # append the loss to the list
        train_losses.append(loss.item())
        # calculate the accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            # set the model to evaluation mode
            net.eval()
            # make the predictions
            y_pred = net(X_test)
            # calculate the loss
            loss = criterion(y_pred, y_test.unsqueeze(1).float())
            # append the loss to the list
            test_losses.append(loss.item())
            # calculate the accuracy
            correct = 0
            total = 0
            # round the predictions
            y_pred = torch.round(y_pred)
            # calculate the accuracy
            correct += (y_pred == y_test.unsqueeze(1)).sum().item()
            total += y_test.size(0)
            # append the accuracy to the list
            test_accuracy.append(correct/total)

        # print the results for every 100 epochs
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracy[-1]:.4f}')
        # update the learning rate
        if epoch % 1000==0:
            lr= lr * lr_decay
    return test_accuracy[-1], net
    
# kfold cross validation
# we will use 5 folds
# we will train the model 5 times
def kfold(X_train, X_test, y_train, y_test, epochs, hyperparameters, lr_decay):
    from sklearn.model_selection import KFold
    kfolds = 5
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    # create the lists for the accuracy
    test_accuracy = []
    
    # train the model 5 times
    for fold, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_train_fold = X_train[train_index]
        X_test_fold = X_train[test_index]
        y_train_fold = y_train[train_index]
        y_test_fold = y_train[test_index]
        print(f'Fold: {fold+1}/{kfolds}')
        print('-----------------------------')
        # train the model
        test_acc, net = train_test(X_train_fold, X_test_fold, y_train_fold, y_test_fold, epochs, hyperparameters, lr_decay)
        # append the accuracy to the list
        test_accuracy.append(test_acc)
        print('-----------------------------')
    # calculate the mean accuracy
    mean_accuracy = np.mean(test_accuracy)
    print(f'Mean accuracy: {mean_accuracy}')
    print('-----------------------------')
    return mean_accuracy, net

#    HYPERPARAMETERS
#^^^^^^^^^^^^^^^^^^^^^^^^
# lr = learning rate
# wd = weight decay
# mm = momentum
# ld = learning rate decay

for lr in {0.001, 0.0001}:
    for wd in {0.0001, 0.00001, 0.000001}:
        for mm in {0, 0.5, 0.9}:
            for lr_decay in {0.9, 0.99}:
                for epochs in {10000}:
                    hyperparameters = {'lr': lr, 'weight_decay': wd, 'momentum': mm}
                    lr_decay = lr_decay
                    # run the model
                    print()
                    print(f'lr: {lr}, wd: {wd}, mm: {mm}, lr_decay: {lr_decay}, epochs: {epochs}')
                    test_accuracy, net = train_test(X_train, X_test, y_train, y_test, epochs, hyperparameters, lr_decay)

                    # csv with colums: epochs, lr, weight_decay, momentum, accuracy
                    # in order to find the best hyperparameters
                    df = pd.read_csv('outputs/model_1/NN_hyperparameters.csv')

                    df = pd.concat([df, pd.DataFrame([[epochs, hyperparameters['lr'], hyperparameters['weight_decay'], \
                                                    hyperparameters['momentum'], lr_decay, test_accuracy]], columns=['epochs', 'lr', \
                                                    'weight_decay', 'momentum', 'lr_decay',  'accuracy'])], axis=0, ignore_index=True)
                    
                    # change the order of the columns
                    df = df[['epochs', 'lr', 'weight_decay', 'momentum', 'lr_decay',  'accuracy']]
                    df.to_csv('outputs/model_1/NN_hyperparameters.csv', index=False, header=True)
                    # save the model's weights in order to plot the features with their weights
                    torch.save(net.state_dict(), 'outputs/model_1/model_1_weights.pth')
                    print('-----------------------------')

print('-----------------------------')
print('-----------------------------')
print('Best model:')
# read the csv with the hyperparameters and run the model with the best accuracy
df = pd.read_csv('outputs/model_1/NN_hyperparameters.csv')
# sort the values by accuracy
df = df.sort_values(by=['accuracy'], ascending=False)
# reset the index
df = df.reset_index(drop=True)
# get the best hyperparameters
epochs = df['epochs'][0]
lr = df['lr'][0]
weight_decay = df['weight_decay'][0]
momentum = df['momentum'][0]
lr_decay = df['lr_decay'][0]
hyperparameters = {'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
# run the model
print()
print(f'lr: {lr}, wd: {weight_decay}, mm: {momentum}, lr_decay: {lr_decay}, epochs: {epochs}')
test_accuracy, net = train_test(X_train, X_test, y_train, y_test, epochs, hyperparameters, lr_decay)

# save the model's weights in order to plot the features with their weights
torch.save(net.state_dict(), 'outputs/model_1/model_1_weights.pth')








                


















