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
count = 0
for i in neurons:
    i = np.array(i)
    i = np.abs(i)
    i = np.mean(i)
    count += 1
    print()
    print(f'The layer {count} has mean weight {i:.3f}')

print()
print('The neural network increases its mean weight as it goes deeper, which means\
that the network is learning to amplify the importance of features or patterns as it goes deeper into the network. ')
print()

# draw the feature space
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import rcParams

# load the test_df
test_df = pd.read_csv('outputs/model_1/test_df.csv')

# PCA to reduce the dimensionality of the data to 2D
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(test_df)
X_pca = pca.transform(test_df)

# plot the data in 2D space 
rcParams['figure.figsize'] = 10, 10

for i in range(11):
    plt.text(X_pca[i, 0], X_pca[i, 1], str(i))
# we will use contourf to plot the decision boundary
# real data's output is at target column and the predicted data's output is at predection column
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=test_df['target'], cmap='coolwarm', marker='o', s=100)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=test_df['prediction'], cmap='coolwarm', marker='x', s=100)
plt.colorbar(ticks = [0,1], label='0 is depressed, 1 is healthy')

plt.title('Feature space')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.savefig('outputs/model_1/feature_space.png')
plt.show()

# patients 0, 1, 4, 10 are the most confused patients by the neural network
# let's see what are the features of these patients
print()
print('The most confused patients are:')
print(test_df.iloc[[0, 1, 4, 10]]) # 8, 21, 50, 16
print()
print('The most confused patients are confused because they have similar features to both classes.\
The neural network is not sure if they are depressed or healthy.')
print()
print(test_df.iloc[[2, 3, 5, 6, 7, 9]]) # 12, 42, 51, 47, 37, 9

#     gender,age,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,target
#8    1,1,1.3255978,-28.30838,17.510078,-3.2645411,17.131327,8.385043,-3.756957,-7.0277934,8.670085,0.677603,0
#21   1,3,-2.8158479,-9.169018,5.8096347,-4.34423,-5.6306696,0.11569762,0.65022874,-0.5651836,-3.1427588,-9.259158,0
#50   1,4,-4.717737,-5.8165827,4.7713842,-6.880367,-2.45959,5.451074,6.69512,3.4010484,-1.4173598,-6.4345074,1
#16   2,8,-1.1351625,-3.2007537,6.2846026,-0.4237101,-5.4102097,10.032163,-8.1067505,-2.5035257,6.1664515,-9.3207245,0

#12   1,6,-7.7129273,-12.955487,10.149863,5.09908,15.603375,8.218982,7.5704985,-6.3059154,1.8839296,-9.803523,0
#42   2,1,-4.4448695,-10.830603,4.0606604,-1.0709376,4.070713,2.315785,10.193716,2.2951405,0.51046026,-11.04419,1
#51   2,7,-2.2065067,-6.3992324,2.4456387,0.49502128,-2.7941546,1.1668297,3.3986568,-3.1596534,-2.7216783,-5.7068152,1
#47   1,1,0.03511405,-12.29185,-1.4109778,-6.423551,-1.1267948,3.724732,2.8224468,4.889318,0.42758405,-8.364201,1
#37   1,7,0.25731182,-9.337095,1.86808,-5.1744027,3.287468,5.4322934,-2.4014652,5.5302043,-3.0607088,-11.060764,1
#9    2,2,11.955008,-3.7961912,21.276836,-11.040052,11.304331,3.2055926,-3.8310769,-2.4706833,15.021927,-16.220394,0

#  So in general wrong predictions  are more likely to be for ages 30-39 and 55-59 (3,4,8),
#  and  also for women (1).

# We know that people with lower "activity measures" are more likely to be depressed.
# And also women (8, 21, 50) or elderly men (16) could not be easily distinguished from the depressed ones of the same age
# and gender, because in general they have a more relaxed lifestyle. On the other hand, men could be more easily distinguished
# since they are more likely to have a more active lifestyle - and if they were depressed, we could easily see the difference.
# Maybe this is also the reason why we have more wrong answers for the people with deppression and less for the healthy/active ones.















