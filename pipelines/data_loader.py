
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load data from the given file path."""
    data = pd.read_csv(file_path)  # Ensure the file is in CSV format (adjust if using other formats)
    return data

def split_data(data, target_column='SalePrice', test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X = data.drop(target_column, axis=1)    # Drop the target column to get features (X)
    y = data[target_column]                 # Extract the target column as the label (y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
