import pandas as pd
import ast
from statistics import mean, stdev

data = pd.read_csv("Data/denoised_action.csv")
data2 = pd.DataFrame()

# mean value of activity for every 30 mins
means = []

# activity for every 1 min of a whole day
max_activity = []
min_activity = []
stds = []

# patient's id
p_id = []

for i in range(0, len(data)):
    activity_list = ast.literal_eval(data.iloc[i]["activity"])
    means.append(mean(activity_list))
    stds.append(stdev(activity_list))
    max_activity.append(max(activity_list))
    min_activity.append(min(activity_list))
    p_id.append(data.iloc[i]["patient"])

data2["means"] = means
data2["stds"] = stds
data2["max"] = max_activity
data2["min"] = min_activity
data2["patient"] = p_id

data2.to_csv("Data/handcraft_features.csv", index=False, header=True)



