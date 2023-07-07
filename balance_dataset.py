def balance_dataset(data):
    # Count the number of samples in each class
    class_counts = data['target'].value_counts()

    # Find the class with the fewest samples
    min_count = class_counts.min()

    # Undersample the majority class
    balanced_df = data.groupby('target').apply(lambda x: x.sample(n=min_count, random_state = 2023)).reset_index(drop=True)

    # Drop the extra data from the majority class
    balanced_df.drop_duplicates(inplace=True)

    return balanced_df
