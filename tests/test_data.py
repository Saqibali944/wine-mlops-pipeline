from src.data import get_data_splits, load_and_validate_data

def test_data_splits_shape():
    X_train, X_test, y_train, y_test = get_data_splits()
    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13
    assert X_train.shape[0] == 142
    assert X_test.shape[0] == 36

def test_no_nulls():
    X, _ = load_and_validate_data()
    assert X.isnull().sum().sum() == 0