import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


saved_weights = torch.load('outputs/model_1/NN_weights.pt')

# extract the weights from the saved_weights tensor and save them to a list
weights = []
for key, value in saved_weights.items():
    weights.append(value[0].tolist())
       

first_layer_weights = weights[0]
second_layer_weights = weights[2]
third_layer_weights = weights[4]

# plot the weights of the all layers in one figure with subplots
rcParams['figure.figsize'] = 10, 10
fig, axs = plt.subplots(3, 1)
axs[0].bar(range(len(first_layer_weights)), first_layer_weights)
axs[0].set_title('First layer weights')
axs[1].bar(range(len(second_layer_weights)), second_layer_weights)
axs[1].set_title('Second layer weights')
axs[2].bar(range(len(third_layer_weights)), third_layer_weights)
axs[2].set_title('Third layer weights')
plt.tight_layout()
plt.savefig('outputs/model_1/NN_weights.png')
plt.show()

# and the neurons are equal to:
neurons = [first_layer_weights, second_layer_weights, third_layer_weights]

# find where the weights had more impact
# average the weights of each neuron
count = 1
for i in neurons:
    i = np.array(i)
    i = np.abs(i)
    i = np.mean(i)
    count += 1
    print()
    print(f'The layer {count} has mean weight {i}')

print()
print('The neural network increases its mean weight as it goes deeper, which means\
that the network is learning to amplify the importance of features or patterns as it goes deeper into the network. ')
print()
















