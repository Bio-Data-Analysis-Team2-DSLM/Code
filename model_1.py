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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# we will add noise to the features and we will create new patients
# loop through the train set
for i in range(0, len(X_train)):
    noise = np.random.normal(0, 0.1, len(X_train.iloc[i]))
    new_patient = X_train.iloc[i] + noise
    new_patient = pd.DataFrame(new_patient).T
    X_train = pd.concat([X_train, new_patient], axis=0, ignore_index=True)
    y_train = pd.concat([y_train, pd.Series([y_train[i]])], axis=0, ignore_index=True)

print()
print(f'Now we have {len(X_train)} patients in the trainig set')
print('-----------------------------')

# convert the data to tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# create the loss function and the optimizer. We will use MSE loss function
criterion = nn.BCELoss() # Binary Cross Entropy loss for binary classification
#optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=0)
# another optimizer that we can use is SGD
################            ################         ################
hyperparameters = {'lr': 0.01, 'weight_decay': 0.0001, 'momentum': 0.8}
################            ################         ################
optimizer = torch.optim.SGD(net.parameters(), **hyperparameters)


# train the model
epochs = 30000
loss_values = []

for i in range(epochs):
    i += 1
    y_pred = net.forward(X_train)

    loss = criterion(y_pred, y_train.unsqueeze(1).float())
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
            g['lr'] = g['lr'] * 0.95

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
    y_val = net.forward(X_test)
    loss = criterion(y_val, y_test.unsqueeze(1).float())
print(f'loss = {loss:.3f}')

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
print()
accuracy = correct/total
print(f'Accuracy: {round(accuracy, 3)*100}%')

# csv with colums: epochs, lr, weight_decay, momentum, accuracy
# in order to find the best hyperparameters
df = pd.read_csv('Data/NN_hyperparameters.csv')
df = pd.concat([df, pd.DataFrame([[epochs, hyperparameters['lr'], hyperparameters['weight_decay'], \
                                    hyperparameters['momentum'], accuracy]], columns=['epochs', 'lr', \
                                    'weight_decay', 'momentum', 'accuracy'])], axis=0, ignore_index=True)
df.to_csv('Data/NN_hyperparameters.csv', index=False, header=True)


# save the model's weights in order to plot the features with their weights
torch.save(net.state_dict(), 'Data/NN_weights.pt')


