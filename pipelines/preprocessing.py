
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib # import joblib for saving the preprocessor

def preprocess_data(X_train, X_test, numerical_features, categorical_features):
    """Preprocess the data: Handle missing values, scale numerical features, and encode categorical features."""
    
    # Numerical Pipeline (Handling missing values and scaling)
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),    # Replace missing values with the mean
        ('scaler', StandardScaler())                    # Scale numerical features
    ])

    # Categorical Pipeline (Handling missing values and encoding)
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing values with the most frequent value
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encoding, pass the sparse_output instead of only sparse
    ])

    # Column Transformer: Apply different transformations to numerical and categorical features
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Fit and transform the training data, then transform the test data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Save the preprocessor after fitting
    joblib.dump(preprocessor, 'pipelines/preprocessor.pkl')  # save the preprocessor to a file.

    return X_train_transformed, X_test_transformed, preprocessor

