import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

####################################################################

def X_features(data):
    data = data.drop(['patient', 'target', 'afftype'], axis=1)

    features = []

    # make patient_new integer
    data['patient_new'] = data['patient_new'].astype(int)

    # create a list with the features of every patient
    for i in data['patient_new'].unique():
        df = data[data['patient_new'] == i]
        df = df.reset_index(drop=True)
        df = df.drop(['patient_new'], axis=1)
        df = df.values
        features.append(df)
        
    # scale the features
    scaler = StandardScaler()
    for i in range(0, len(features)):
        features[i] = scaler.fit_transform(features[i])

    # make the lists of the same length
    for i in range(0, len(features)):
        features[i] = features[i].reshape(-1, 1)
        

    # convert the lists to tensors
    for i in range(0, len(data['patient_new'].unique())):
        features[i] = torch.tensor(features[i], dtype=torch.float32)

    # create the X tensors
    X = torch.stack(features)

    # reshape the tensors to have shape (new_patients, 1, 48)
    X = X.view(len(data['patient_new'].unique()), 1, 48)

    return X


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
    
    # define the forward pass using self.x
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# in order to have the same results every time we run the code
torch.manual_seed(88)
np.random.seed(44)

def CNN_train_test(a, X_train, X_test, name):
    # set the seeds
    torch.manual_seed(24)
    np.random.seed(42)
    # define the model
    model = CNN(a)
    print(f'Training model {name}...')

    # define the loss function
    loss_function = nn.MSELoss()
    # define the optimizer - Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # train the model
    epochs = 10000
    loss_values = []

    for i in range(epochs):
        i += 1
        y_pred = model.forward(X_train)
        loss = loss_function(y_pred, X_train)
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
    plt.savefig(f'Data/loss_CNN_'+ name +'png')

    # test the model
    with torch.no_grad():
        y_val = model.forward(X_test)
        loss = loss_function(y_val, X_test)

    print(f'loss on the test set = {loss:.3f}')

    return model



def extract_features(model, X, name):
    print(f'Extracting features for {name}...')
    features = model.encoder(X).detach().numpy()
    features = features.reshape(-1, model.a)
    # create a dataframe with headers f1, f2, ..., f10
    headers = ['f' + str(i) for i in range(1, model.a + 1)]
    # add the headers to the dataframe
    features = pd.DataFrame(features, columns=headers)
    features.to_csv('Data/extracted_features_for_'+name+'.csv', index=False, header=True)

train1 = pd.read_csv('Data/train1.csv')
test1 = pd.read_csv('Data/test1.csv')
train2 = pd.read_csv('Data/train2.csv')
test2 = pd.read_csv('Data/test2.csv')

X_train1 = X_features(train1)
X_test1 = X_features(test1)
X_train2 = X_features(train2)
X_test2 = X_features(test2)

model_1 = CNN_train_test(30, X_train1, X_test1, 'model1')
model_2 = CNN_train_test(30, X_train2, X_test2, 'model2')

extract_features(model_1, X_train1, 'train1')
extract_features(model_1, X_test1, 'test1')
extract_features(model_2, X_train2, 'train2')
extract_features(model_2, X_test2, 'test2')




