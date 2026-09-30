import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def load_and_validate_data():
    """Loads the Wine dataset and runs validation checks."""
    data = load_wine(as_frame=True)
    X = data.data
    y = data.target

    # Validation Check 1: Feature count must equal 13
    assert X.shape[1] == 13, f"Expected 13 features, got {X.shape[1]}"
    
    # Validation Check 2: No null values exist
    assert X.isnull().sum().sum() == 0, "Dataset contains null values"

    return X, y

def get_data_splits():
    """Performs a stratified 80/20 train-test split."""
    X, y = load_and_validate_data()
    
    # Stratified 80/20 split using random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Quick test when you execute this specific file
    X_train, X_test, y_train, y_test = get_data_splits()
    print("Data validation passed!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")