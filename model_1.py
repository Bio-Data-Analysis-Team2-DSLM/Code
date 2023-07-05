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


data = pd.DataFrame()

df_scores_1 = pd.read_csv('Data/scores_for classification_1.csv')

df_features = pd.read_csv('Data/extracted_features.csv')

# merge the scores with the features
dataset = pd.concat([df_scores_1, df_features], axis=1)

# move the target column to the end of the dataframe
cols = list(dataset.columns)
dataset = dataset[cols[0:2]+cols[3:]+[cols[2]]]

# split to features and targets
y = dataset['target']
X = dataset.drop(['target'], axis=1)

# rescale the features
scaler = StandardScaler()
scaler.fit(X['age'].values.reshape(-1, 1))
X['age'] = scaler.transform(X['age'].values.reshape(-1, 1))
scaler.fit(X['gender'].values.reshape(-1, 1))
X['gender'] = scaler.transform(X['gender'].values.reshape(-1, 1))
for i in range(1, 11):
    scaler.fit(X[f'f{i}'].values.reshape(-1, 1))
    X[f'f{i}'] = scaler.transform(X[f'f{i}'].values.reshape(-1, 1))



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
        self.fc1 = nn.Linear(12, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # we have classification problem so we will use sigmoid function
        x = torch.sigmoid(self.fc3(x))
        return x

# set the seeds
torch.manual_seed(42)
np.random.seed(42)

# create the model
net = Net()

# split the data to train and test and reset the indices
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# we will add noise to the features and we will create new patients
# loop through the train set


for i in range(0, len(X_train)):
    for j in range(1):
        noise = np.random.normal(0, 1.2, len(X_train.iloc[i])-2)

        X_train.iloc[i][2:] = X_train.iloc[i][2:] + noise
        new_patient = X_train.iloc[i]
        new_patient = pd.DataFrame(new_patient).T

        X_train = pd.concat([X_train, new_patient], axis=0, ignore_index=True)
        y_train = pd.concat([y_train, pd.Series([y_train[i]])], axis=0, ignore_index=True)




print()
print(f'Now we have {len(X_train)} patients in the trainig set')
print(f'And {len(X_test)} patients in the test set')
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

################            ################         ################

hyperparameters = {'lr': 1e-2, 'weight_decay': 1e-5, 'momentum': 0.5}
lr_decay = 0.9   # 1 is for no decay

################            ################         ################

optimizer = torch.optim.SGD(net.parameters(), **hyperparameters)


# train the model
epochs = 60000
loss_values = []

for i in range(epochs):
    i += 1
    y_pred = net.forward(X_train_)

    loss = criterion(y_pred, y_train_.unsqueeze(1).float())
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

# plot the loss
import matplotlib.pyplot as plt
x = [i[0] for i in loss_values]
y = [i[1] for i in loss_values]
plt.plot(x, y)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# test the model
with torch.no_grad():
    y_val = net.forward(X_test_)
    loss = criterion(y_val, y_test_.unsqueeze(1).float())

# calculate the accuracy
correct = 0
total = 0
correct_predictions = []
with torch.no_grad():
    predictions = net.forward(X_test_)
    for i in range(len(y_test_)):
        if predictions[i] >= 0.5:
            y_pred = 1
        else:
            y_pred = 0
        if y_pred == y_test_[i]:
            correct += 1
            print(f'correct prediction: {i}')
            correct_predictions.append(i)
        total += 1
print()
accuracy = correct/total
print(f'loss on the original test set = {loss:.3f}')
print(f'Accuracy on the original test set: {round(accuracy, 4)*100:3.4f}%')


# We also want to check if our model is sensitive to small changes in the input
# because we use momentum in the optimizer, which could make our model more robust
# to small changes in the input

# loop through the test set
for i in range(0, len(X_test)):
    for j in range(1):
        noise = np.random.normal(0, 1, len(X_test.iloc[i])-2)

        X_test.iloc[i][2:] = X_test.iloc[i][2:] + noise
        new_patient = X_test.iloc[i]
        new_patient = pd.DataFrame(new_patient).T

        X_test = pd.concat([X_test, new_patient], axis=0, ignore_index=True)
        y_test = pd.concat([y_test, pd.Series([y_test[i]])], axis=0, ignore_index=True)

# convert the data to tensors
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# test the model
with torch.no_grad():
    y_val = net.forward(X_test)
    loss2 = criterion(y_val, y_test.unsqueeze(1).float())
print()
print(f'loss of the noisy test set = {loss2:.3f}')

# calculate the accuracy
correct = 0
total = 0
with torch.no_grad():
    predictions = net.forward(X_test)
    for i in range(len(y_test)):
        if predictions[i] >= 0.5:
            y_pred = 1
        else:
            y_pred = 0
        if y_pred == y_test[i]:
            correct += 1
        total += 1

accuracy2 = correct/total
print(f'Accuracy on the noisy test set: {round(accuracy2, 4)*100:3.4f}%')
print()

# check if the accurasy of the noisy test set is lower than the 0.8*accuracy of the original test set
if accuracy2 < 0.92*accuracy:
    print('The model is sensitive to small changes in the input')
    a = 'fail'
else:
    print('The model is not sensitive to small changes in the input')
    a = 'pass'


# csv with colums: epochs, lr, weight_decay, momentum, accuracy
# in order to find the best hyperparameters
df = pd.read_csv('outputs/model_1/NN_hyperparameters.csv')



df = pd.concat([df, pd.DataFrame([[epochs, hyperparameters['lr'], hyperparameters['weight_decay'], \
                                    hyperparameters['momentum'], lr_decay, correct_predictions, accuracy, a]], columns=['epochs', 'lr', \
                                    'weight_decay', 'momentum', 'lr_decay', 'correct_predictions', 'accuracy', 'sensitivity'])], axis=0, ignore_index=True)

# change the order of the columns
df = df[['epochs', 'lr', 'weight_decay', 'momentum', 'lr_decay', 'correct_predictions', 'accuracy', 'sensitivity']]

df.to_csv('outputs/model_1/NN_hyperparameters.csv', index=False, header=True)


# save the model's weights in order to plot the features with their weights
torch.save(net.state_dict(), 'outputs/model_1/NN_weights.pt')


# merge X_test with y_test and save it to csv
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)

# merge X_test_df with y_test_df
test_df = pd.concat([X_test_df, y_test_df], axis=1)

# save the test_df to csv
test_df.to_csv('outputs/model_1/test_df.csv', index=False, header=True)
