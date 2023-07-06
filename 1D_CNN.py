import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Data/action_cnn.csv')
first_row = pd.read_csv('Data/first_row_for_each_patient.csv')

data = pd.DataFrame(data)
data = data.drop(['patient'], axis=1)

# create a new column which will assign the same number to every 48 rows
data['patient'] = 0
for i in range(0, 49392, 48):
    data.loc[i:i+48, 'patient'] = i/48 + 1

# find the fist patient with target 1
for i in range(0, 49392, 48):
    if data['target'][i] == 1:
        break
print()
print(f'first patient with target 1 is patient #{data["patient"][i]} at row {i}')
print()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# κωδικας για real patient number

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

features = []
# create a list with the features of every patient
# number of patients = 1029  and number of features = 48
for i in range(1, 1030):
    df = data[data['patient'] == i]
    df = df.reset_index(drop=True)
    df = df.drop(['patient', 'target'], axis=1)
    df = df.values
    features.append(df)

# normalize the data to have mean 0 and std 1 for every list of features 
for i in range(0, 1029):
    scaler = StandardScaler()
    features[i] = scaler.fit_transform(np.array(features[i]).reshape(-1, 1))
# make the lists of the same length
for i in range(0, 1029):
    features[i] = features[i].reshape(-1, 1)

# convert the lists to tensors
for i in range(0, 1029):
    features[i] = torch.tensor(features[i], dtype=torch.float32)

# create the X tensors
X = torch.stack(features)

# reshape the tensors to have shape (55, 1, 546)
X = X.view(1029, 1, 48)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# we create a model of 1D CNN with pytorch which will extract features from the data
# so that we use the extracted featurea in a NN for classification
class CNN(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.Linear(48, a))
        
        self.decoder = nn.Sequential(
            nn.Linear(a, 48),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=1))


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# in order to have the same results every time we run the code
torch.manual_seed(88)
np.random.seed(44)

# define the model
model = CNN(30)

# set the initial weights of the model
def init_weights(m):
    if type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

# define the loss function
loss_function = nn.MSELoss()
# define the optimizer - Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# train the model
epochs = 20000
loss_values = []

for i in range(epochs):
    i += 1
    y_pred = model.forward(X)
    loss = loss_function(y_pred,X)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        # decrease the learning rate every 300 epochs
    if i % 1000 == 0:
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * 0.9

    # print every 1000 epochs
    if i % 1000 == 0:
        print(f'epoch: {i:2} loss: {loss.item():10.3f}')
        # save the loss values as a list [epochs, loss] in order to plot them
        loss_values.append([i, loss.item()])

     
# plot the loss values
import matplotlib.pyplot as plt
x = [i[0] for i in loss_values]
y = [i[1] for i in loss_values]
plt.plot(x, y)
plt.show()
# save the loss picture
plt.savefig('Data/loss_values_CNN.png')

# save the extracted features as a new dataframe
features = model.encoder(X).detach().numpy()
features = features.reshape(-1, model.a)
# create a dataframe with headers f1, f2, ..., f10
headers = ['f' + str(i) for i in range(1, model.a + 1)]
# add the headers to the dataframe
features = pd.DataFrame(features, columns=headers)

features.to_csv('Data/extracted_features.csv', index=False, header=True)

# print loss
print(f"loss = {loss.item()}")
