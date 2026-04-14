import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    # 1. Separate features and target
    X = df.drop("condition", axis=1)  
    y = df["condition"]

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Scaling (only features)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# test
if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Preprocessing done")
    print(X_train.shape, X_test.shape)