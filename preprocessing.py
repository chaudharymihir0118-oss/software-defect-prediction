import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file):
    df = pd.read_csv(file)
    return df

def preprocess_data(df):
    # Handle missing values
    df = df.fillna(df.mean())

    X = df.drop("defects", axis=1)
    y = df["defects"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)