import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import pandas as pd
from data import get_data_splits

def train_and_log():
    # 1. Setup MLflow using SQLite so the Model Registry works locally
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "Wine-Cultivar-Classification"
    mlflow.set_experiment(experiment_name)

    # 2. Load Data
    X_train, X_test, y_train, y_test = get_data_splits()

    # 3. Define 2 Model Families with 3 configurations each
    experiments = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": [
                {"n_estimators": 50, "max_depth": 3},
                {"n_estimators": 100, "max_depth": 5},
                {"n_estimators": 150, "max_depth": None}
            ]
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": [
                {"n_estimators": 50, "learning_rate": 0.05},
                {"n_estimators": 100, "learning_rate": 0.1},
                {"n_estimators": 150, "learning_rate": 0.2}
            ]
        }
    }

    # Setup 5-fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['f1_macro', 'accuracy', 'neg_log_loss']

    # 4. Train and Track Experiments
    for model_name, config in experiments.items():
        base_model = config["model"]
        for params in config["params"]:
            with mlflow.start_run(run_name=f"{model_name}_{params['n_estimators']}"):
                model = base_model.set_params(**params)
                
                # Run Cross-validation
                cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
                
                # Calculate mean metrics
                mean_f1 = cv_results['test_f1_macro'].mean()
                mean_acc = cv_results['test_accuracy'].mean()
                mean_log_loss = -cv_results['test_neg_log_loss'].mean()

                # Log to MLflow
                mlflow.log_param("model_family", model_name)
                mlflow.log_params(params)
                mlflow.log_metric("val_f1_macro", mean_f1)
                mlflow.log_metric("val_accuracy", mean_acc)
                mlflow.log_metric("val_log_loss", mean_log_loss)

                # Fit on full training split to save the artifact
                model.fit(X_train, y_train)

                # Infer signature (input/output schema) and log the model
                signature = infer_signature(X_train, model.predict(X_train))
                input_example = X_train.iloc[[0]]
                
                # FIXED: Added serialization_format to bypass skops security error
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                    serialization_format="cloudpickle" 
                )
    
    # 5. Model Registry and Champion Promotion
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    # Search MLflow for the absolute best run based on Macro F1
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_f1_macro DESC"],
        max_results=1
    )
    
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    print(f"Best Run ID: {best_run_id} with F1-Score: {best_run.data.metrics['val_f1_macro']:.4f}")

    # Register the winning model
    model_name = "WineClassifier"
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(model_uri, model_name)
    
    # Assign the "champion" alias
    client.set_registered_model_alias(model_name, "champion", model_version.version)
    print(f"Successfully registered version {model_version.version} as 'champion'!")

if __name__ == "__main__":
    train_and_log()