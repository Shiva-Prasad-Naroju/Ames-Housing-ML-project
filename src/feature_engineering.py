import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso


# Encode the category columns, so just passthe cat_cols
def binary_encode_features(X_train, X_test):
    """
    Dynamically applies Binary Encoding to categorical columns in X_train and X_test.

    Args:
    X_train (pd.DataFrame): Training dataset.
    X_test (pd.DataFrame): Test dataset.

    Returns:
    pd.DataFrame, pd.DataFrame: Encoded X_train and X_test.
    """
    # Dynamically identify categorical columns in X_train
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Initialize the BinaryEncoder
    encoder = ce.BinaryEncoder(cols=cat_cols)

    # Apply Binary Encoding to the training dataset
    X_train_encoded = encoder.fit_transform(X_train[cat_cols])

    # Apply the same transformation to the test dataset
    X_test_encoded = encoder.transform(X_test[cat_cols])

    # Drop the original categorical columns and add the encoded columns
    X_train = X_train.drop(columns=cat_cols)
    X_test = X_test.drop(columns=cat_cols)

    # Concatenate the encoded columns with the original ones (if any)
    X_train = pd.concat([X_train, X_train_encoded], axis=1)
    X_test = pd.concat([X_test, X_test_encoded], axis=1)

    return X_train, X_test


# apply scaling the numerical features using Standard Scaler:
def scale_num_cols_using_StandardScaler(X_train, X_test):
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()

    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test


# Scales numerical columns in X_train and X_test using Min-Max Scaling.
def scale_num_cols_using_MinMaxScaler(X_train, X_test):
    from sklearn.preprocessing import MinMaxScaler
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    scaler = MinMaxScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test


# Feature selection using Lasso Regression.
def select_features_lasso(X_train, X_test, y_train, alpha=0.1, random_state=43):
    # Feature selection using Lasso
    selector = SelectFromModel(Lasso(alpha=alpha, random_state=random_state))
    selector.fit(X_train, y_train)

    # Get the selected features
    selected_features = X_train.columns[selector.get_support()]

    # Apply the selected features to both X_train and X_test
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    print(f'Selected features: {len(selected_features)}')
    print(f'Features with coefficients shrank to zero: {np.sum(selector.estimator_.coef_ == 0)}')
    print(f'X_train shape after feature selection: {X_train.shape}')
    print(f'X_test shape after feature selection: {X_test.shape}')

    return X_train, X_test


