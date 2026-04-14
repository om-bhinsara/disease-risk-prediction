from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "ROC-AUC": roc_auc
        }

        print(f"\n{name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

    return results


# test
if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_data
    from train import train_models

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = train_models(X_train, y_train)

    results = evaluate_models(models, X_test, y_test)
    
    ''' 
    
    Model Selection

    Among the three models (Logistic Regression, Random Forest, and SVM), Random Forest performed the best across all evaluation metrics. It achieved the highest accuracy (76.67%) and ROC-AUC score (0.8538), indicating better overall performance in distinguishing between classes.

    Additionally, Random Forest handles non-linear relationships and is robust to outliers, making it well-suited for this dataset. Therefore, it was selected as the final model.
    
    '''