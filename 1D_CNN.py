import pandas as pd

data = pd.read_csv('https://www.kaggle.com/datasets/arashnic/the-depression-dataset')

X = data.drop('target', axis=1)
y = data['target']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 2)
        self.fc1 = nn.Linear(32 * (X_train.shape[1] - 1), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-, 32 * (X_train.shape[1] - 1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

