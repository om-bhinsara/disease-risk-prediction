from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def train_models(X_train, y_train):
    models = {}

    # 1. Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr

    # 2. Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    # 3. SVM (with probability)
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    models["SVM"] = svm

    return models


# test
if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_data

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = train_models(X_train, y_train)

    print("Models trained:")
    for name in models:
        print(name)