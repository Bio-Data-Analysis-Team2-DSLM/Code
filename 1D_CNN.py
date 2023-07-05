import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Data/data_for_feat_extr.csv')
# create dataframe for data
data = pd.DataFrame(data)

# create a list of lists with the features
features = []
for i in range(0, 55):
    features.append(data['activity'][i])

# convert string "['1', '2', '3', ...]" to list [1, 2, 3, ...]
for i in range(0, 55):
    features[i] = features[i].replace("[", "")
    features[i] = features[i].replace("]", "")
    features[i] = features[i].replace("'", "")
    features[i] = features[i].split(", ")
    features[i] = list(map(int, features[i]))

# simplify features by taking the mean of every 100 values in a list
for i in range(0, 55):
    features[i] = [sum(features[i][j:j+120])/120 for j in range(0, len(features[i]), 120)]

# the lists are not of the same length
# so we need to pad them with the mean of the list

# we need to find the maximum length of the lists
max_len = 0
for i in range(0, 55):
    if len(features[i]) > max_len:
        max_len = len(features[i])

# pad the lists with the mean of the list for the rest of the values
for i in range(0, 55):
    if len(features[i]) < max_len:
        # if the value is nan for the rest of the values then replace it with the mean of the list
        # just for the values that are nan, not for the whole list
        for isnan in range(0, max_len-len(features[i])):
            mean = sum(features[i])/len(features[i])
            features[i].append(mean)

# normalize the data to have mean 0 and std 1 for every list of features 
for i in range(0, 55):
    scaler = StandardScaler()
    features[i] = scaler.fit_transform(np.array(features[i]).reshape(-1, 1))
# make the lists of the same length
for i in range(0, 55):
    features[i] = features[i].reshape(546)

# convert the lists to tensors
for i in range(0, 55):
    features[i] = torch.tensor(features[i], dtype=torch.float32)

# create the X tensors
X = torch.stack(features)

# reshape the tensors to have shape (55, 1, 546)
X = X.view(55, 1, 546)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# we create a model of 1D CNN with pytorch which will extract features from the data
# so that we use the extracted featurea in a NN for classification
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.Linear(546, 10))
        
        self.decoder = nn.Sequential(
            nn.Linear(10, 546),
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
model = CNN()

# set the initial weights of the model
def init_weights(m):
    if type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

# define the loss function
loss_function = nn.MSELoss()
# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

# train the model
epochs = 12000
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

    # print every 20000 epochs
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
plt.savefig('Data/loss_values_CNN.jpg')

# save the extracted features as a new dataframe
features = model.encoder(X).detach().numpy()
features = features.reshape(-1, 10)
# create a dataframe with headers f1, f2, ..., f10
headers = ['f' + str(i) for i in range(1, 11)]
# add the headers to the dataframe
features = pd.DataFrame(features, columns=headers)
features.to_csv('Data/extracted_features.csv', index=False, header=True)

# print loss
print(f"loss = {loss.item()}")
