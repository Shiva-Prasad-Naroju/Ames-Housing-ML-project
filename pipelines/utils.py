
import joblib
import logging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)

def save_model(model, file_path):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)
    logging.info(f"Model saved to {file_path}")

def load_model(file_path):
    """Load a saved model from a file."""
    model = joblib.load(file_path)
    logging.info(f"Model loaded from {file_path}")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    # Make predictions using the trained model
    predictions = model.predict(X_test)
    
    # Calculate and log evaluation metrics
    mae = mean_absolute_error(y_test,predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    logging.info(f"Mean Absolute Error: {mae}")
    logging.info(f"Mean Squared Error: {mse}")
    logging.info(f"R-Squared: {r2}")
    
    return mae, mse, r2

