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


data = pd.DataFrame()

#df_scores_1 = pd.read_csv('Data/scores_for classification_1.csv')
df_features = pd.read_csv('Data/extracted_features.csv')
# add the target column to the features
df_features['target'] = 0
# assign the first 359 patients (17232/48) to 0 and the rest to 1
for i in range(0, 359):
    df_features.loc[i, 'target'] = 0
for i in range(359, 1029):
    df_features.loc[i, 'target'] = 1

dataset = df_features
# save the dataset to csv
dataset.to_csv('Final/dataset_1.csv', index=False)

counts = dataset['target'].value_counts()
print(f'Before balancing :{counts}')
# balance the dataset
dataset = balance_dataset.balance_dataset(dataset)
counts = dataset['target'].value_counts()
print(f'After: {counts}')

# scale the data from all the features except the target
scaler = StandardScaler()
dataset.iloc[:, 1:-1] = scaler.fit_transform(dataset.iloc[:, 1:-1])

# split to features and targets
y = dataset['target']
X = dataset.drop(['target'], axis=1)

# save X and y in one csv
dataset.to_csv('Data/dataset_before_scaling.csv', index=False)


# save X and y in one csv
dataset = pd.concat([X, y], axis=1)
dataset.to_csv('Data/dataset_after_scaling.csv', index=False)

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



# split the data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
X_train_ = torch.tensor(X_train.values, dtype=torch.float32)
X_test_ = torch.tensor(X_test.values, dtype=torch.float32)
y_train_ = torch.tensor(y_train.values, dtype=torch.float32)
y_test_ = torch.tensor(y_test.values, dtype=torch.float32)



# create the loss function and the optimizer. We will use MSE loss function
criterion = nn.BCELoss() # Binary Cross Entropy loss for binary classification
#optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=0)
# another optimizer that we can use is SGD

# run model_1.py for different hyperparameters 
# and save the results in a csv file
# lr = learning rate
# wd = weight decay
# mm = momentum
# ld = learning rate decay


# best run for 10000,0.0001,0.0001,0.99,0.9
for lr in [ 0.001, 0.0001]:
    for wd in [ 0.0000001]:
        for mm in [ 0.5]:
            for ld in [0.9]:
                hyperparameters = {'lr': lr, 'weight_decay': wd, 'momentum': mm}
                lr_decay = ld
                print(f'lr = {lr}, wd = {wd}, mm = {mm}, ld = {ld}')
                

                hyperparameters = {'lr': lr, 'weight_decay': wd, 'momentum': mm}
                lr_decay = ld

                


                # train the model
                epochs = 10000
                loss_values = []

                from sklearn.model_selection import KFold

                # Define the number of folds for cross-validation
                num_folds = 5

                # Create a KFold object
                kf = KFold(n_splits=num_folds, shuffle=True)
                for fold, (train_ids, val_ids) in enumerate(kf.split(X_train_)):
                    X_train_fold = X_train_[train_ids]
                    y_train_fold = y_train_[train_ids]
                    X_val_fold = X_train_[val_ids]
                    y_val_fold = y_train_[val_ids]

                    net = Net()
                    optimizer = torch.optim.SGD(net.parameters(), **hyperparameters)

                    for i in range(epochs):
                        i += 1
                        y_pred = net.forward(X_train_fold)

                        loss = criterion(y_pred, y_train_fold.unsqueeze(1).float())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # print every 10 epochs
                        if i%1000 == 0:
                            print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
                            loss_values.append([i, loss.item()])
                        # decrease the learning rate every 300 epochs
                        if i % 1000 == 0:
                            for g in optimizer.param_groups:
                                g['lr'] = g['lr'] * lr_decay



                # test the model
                with torch.no_grad():
                    y_val = net.forward(X_test_)
                    loss = criterion(y_val, y_test_.unsqueeze(1).float())

                # calculate the accuracy
                correct = 0
                total = 0
                correct_predictions = []
                prediction = []
                with torch.no_grad():
                    predictions = net.forward(X_test_)
                    for i in range(len(y_test_)):
                        if predictions[i] >= 0.5:
                            y_pred = 1
                            prediction.append(1)
                        else:
                            y_pred = 0
                            prediction.append(0)
                        if y_pred == y_test_[i]:
                            correct += 1
                            print(f'correct prediction: {i}')
                            correct_predictions.append(i)
                        total += 1
                print()
                accuracy = correct/total
                print(f'loss on the test set = {loss:.3f}')
                print(f'Accuracy on the test set: {round(accuracy, 4)*100:3.4f}%')

                # plot the loss
                import matplotlib.pyplot as plt
                x = [i[0] for i in loss_values]
                y = [i[1] for i in loss_values]
                plt.plot(x, y)
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.show()
                

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


                # merge X_test with y_test and save it to csv
                X_test_df = pd.DataFrame(X_test_)
                y_test_df = pd.DataFrame(y_test_)
                predictions_df = pd.DataFrame(prediction)
                # merge X_test_df with y_test_df
                test_df = pd.concat([X_test_df, y_test_df, predictions_df], axis=1, ignore_index=True)

                # header
                header = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', \
                        'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', \
                            'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', \
                                'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'target', 'prediction']
                
                

                # save the test_df to csv
                test_df.to_csv('outputs/model_1/test_df.csv', index=False, header=header)






