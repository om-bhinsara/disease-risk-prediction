import pickle
import os

def save_model(model, filename="models/model.pkl"):
    # Create folder if not exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as file:
        pickle.dump(model, file)

    print("Model saved successfully")
    
def save_scaler(scaler, filename="models/scaler.pkl"):
    import os, pickle
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as file:
        pickle.dump(scaler, file)

    print("Scaler saved successfully")
    
if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_data
    from train import train_models
    from evaluate import evaluate_models

    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    models = train_models(X_train, y_train)

    results = evaluate_models(models, X_test, y_test)

    # Select best model
    best_model = models["Random Forest"]

    # Save best model
    save_model(best_model)
    
    # Save scaler
    save_scaler(scaler)