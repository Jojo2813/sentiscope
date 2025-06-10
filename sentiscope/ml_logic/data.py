import os
import bz2
import pandas as pd

def load_data(data_path):
    """
    Function that
    - loads data from csv if the csv exists
    - loads data from .bz2 file and stores it as csv otherwise

    Input: path to raw data folder (not the files directly)
    Output: Training DataFrame and Testing DataFrame

    """

    def parse_raw_files(file_path):
        """
        Function that loads data out of a .bz2 file and converts it into a pandas
        DataFrame
        """

        data = []

        with bz2.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ' , 1)
                if len(parts) == 2:
                    label, text = parts
                    label = label.replace('__label__', '')
                    data.append((label, text))

        df = pd.DataFrame(data, columns=['label', 'text'])
        df['label'] = df['label'].astype(int)

        return df

    #Path to csv data files
    train_path = os.path.join(data_path, "train.ft.txt.bz2")
    test_path = os.path.join(data_path, "test.ft.txt.bz2")

    # Try to load data from csv, otherwise load from .bz2 and store as csv
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        print("Training data loaded from csv")
    else:
        # Your alternative logic here
        train_df = parse_raw_files(train_path)
        print("Training data loaded from bz2 and stored as csv")

    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print("Testing data loaded from csv")
    else:
        # Your alternative logic here
        test_df = parse_raw_files(test_path)
        print("Testing data loaded from bz2 and stored as csv")

    return train_df, test_df
