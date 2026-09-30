import mlflow
from sklearn.metrics import f1_score, accuracy_score
from data import get_data_splits

def evaluate_model():
    # 1. Point MLflow to our local database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # 2. Load the test data (we only need X_test and y_test)
    _, X_test, _, y_test = get_data_splits()
    
    # 3. Load the 'champion' model directly from the MLflow Registry
    model_uri = "models:/WineClassifier@champion"
    print(f"Loading champion model from registry: {model_uri}...")
    model = mlflow.pyfunc.load_model(model_uri)
    
    # 4. Perform inference (predictions)
    print("Running inference on test split...")
    predictions = model.predict(X_test)
    
    # 5. Calculate and print final metrics
    macro_f1 = f1_score(y_test, predictions, average='macro')
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n" + "="*30)
    print(" CHAMPION MODEL TEST METRICS")
    print("="*30)
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Accuracy:       {accuracy:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_model()