import pandas as pd


def denoised_data(file, outputfile):
    data = pd.read_csv(file)
    denoised_data = pd.DataFrame()

    # mean value of activity for every 30 mins
    means = []

    # patient's id
    p_id = []

    # match every day to a new patient's id
    for i in range(0, len(data), 30):
        a = data.iloc[i:i + 30]["activity"].mean()
        means.append(a)
        p_id.append(data.iloc[i]["patient"])

    denoised_data["activity"] = means
    denoised_data["patient"] = p_id

    final_values = []
    # patient's id
    p_id = []

    for i in range(0, len(denoised_data), 48):
        a = denoised_data.iloc[i:i + 48]["activity"]
        final_values.append(list(a))
        p_id.append(denoised_data.iloc[i]["patient"])

    denoised_data = pd.DataFrame()
    denoised_data["activity"] = final_values
    denoised_data["patient"] = p_id

    # save data
    denoised_data.to_csv(outputfile, index=False, header=True)


denoised_data('Data/action.csv', 'Data/denoised_action.csv')
