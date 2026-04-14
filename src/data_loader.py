import pandas as pd

def load_data():
    df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Videos\Code Alpha Internship\Project 2\data\heart_cleveland_upload.csv")
    return df


if __name__ == "__main__":
    data = load_data()
    print(data.head())