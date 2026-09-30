import time
import mlflow
from sklearn.metrics import f1_score
from src.data import get_data_splits

def load_champion_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = "models:/WineClassifier@champion"
    return mlflow.pyfunc.load_model(model_uri)

def test_model_f1_score_gate():
    model = load_champion_model()
    _, X_test, _, y_test = get_data_splits()
    
    predictions = model.predict(X_test)
    macro_f1 = f1_score(y_test, predictions, average='macro')
    
    assert macro_f1 >= 0.88, f"Model failed F1 gate: {macro_f1}"

def test_inference_latency_gate():
    model = load_champion_model()
    _, X_test, _, _ = get_data_splits()
    
    model.predict(X_test) # Warmup
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000
    assert latency_ms <= 30.0, f"Model failed latency gate: {latency_ms} ms"

def test_output_schema_integrity():
    model = load_champion_model()
    _, X_test, _, _ = get_data_splits()
    
    predictions = model.predict(X_test)
    unique_preds = set(predictions)
    
    allowed_classes = {0, 1, 2}
    assert unique_preds.issubset(allowed_classes), f"Invalid predictions: {unique_preds}"