# import the libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# load the data
data = pd.read_csv('Data/dataset_after_scaling.csv')
scores = pd.read_csv('Data/scores_for classification_2.csv')

scores = scores.drop(['target', 'gender','age'], axis=1)
data = data.drop(['target'], axis=1)

# scale the scores
scaler = StandardScaler()
scaler.fit(scores['inpatient'].values.reshape(-1, 1))
scores['inpatient'] = scaler.transform(scores['inpatient'].values.reshape(-1, 1))

# add the scores to the data dataframe and save it to a csv
data = pd.concat([scores, data], axis=1)
data.to_csv('Data/dataset_2_after_scaling.csv', index=False)

# split to features and targets
y = data['afftype']
X = data.drop(['afftype'], axis=1)

print(X.head())

# split to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# convert the data to tensors
X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)
y_train = torch.LongTensor(y_train.values)
y_test = torch.LongTensor(y_test.values)


# Create a Neural Network that will take the data and will classify them
# to afftype 1 or 2

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 1)
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # we have classification problem so we will use sigmoid function
        x = torch.sigmoid(self.fc3(x))
        return x
    
model = Net()
print(model)

# define the loss function and the optimizer. we have classification problem
# so we will use Binary Cross Entropy Loss
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train the model
epochs = 100
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train.unsqueeze(1).float())
    losses.append(loss)
    
    if i % 10 == 0:
        print(f'epoch {i} and loss is: {loss}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# plot the loss function
import matplotlib.pyplot as plt
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')

# save the model
torch.save(model.state_dict(), 'outputs/model_2/NN_model_2.pt')

# evaluate the model
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test.unsqueeze(1).float())
print(f'Loss: {loss:.8f}')

# get the predictions
predictions = []
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_pred = model(data)
        predictions.append(y_pred.argmax().item())

# create a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print(cm)

# plot the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.3f}')

# calculate the precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, predictions)
print(f'Precision: {precision:.3f}')

# calculate the recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, predictions)
print(f'Recall: {recall:.3f}')

# calculate the f1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_test, predictions)
print(f'F1 score: {f1:.3f}')




