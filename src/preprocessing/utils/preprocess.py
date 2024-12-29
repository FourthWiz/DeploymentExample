"""
Module for preprocessing the data.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

TEXT_COLUMNS = [
    "native-country",
    "sex",
    "relationship",
    "occupation",
    "education",
    "workclass",
    "marital-status",
    "race"
]

def process_text_column(data, column):
    data[column] = data[column].str.strip()
    return data

def label_encode(data, columns):
    label_encoders = {}

    for col in columns:
        le = LabelEncoder()
        data.loc[:, col] = le.fit_transform(data[col])
        label_encoders[col] = le

    return data, label_encoders


def preprocess_data(path):
    """
    This function preprocesses the data.
    """
    df = pd.read_csv(path)
    df.columns = [x.strip() for x in df.columns]
    for col in TEXT_COLUMNS:
        df = process_text_column(df, col)

    df = process_text_column(df, 'income')
    df.loc[df['income'].str.contains('<=50K'), 'income'] = '0'
    df.loc[df['income'].str.contains('>50K'), 'income'] = '1'
    df.loc[:, 'income'] = df['income'].astype(int)
    df.drop_duplicates(inplace=True)

    return df
