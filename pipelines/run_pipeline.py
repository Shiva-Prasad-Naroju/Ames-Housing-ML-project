
import logging
logging.basicConfig(level=logging.INFO)

from data_loader import load_data, split_data
from preprocessing import preprocess_data
from model_training import train_model, evaluate_model
from utils import save_model,evaluate_model

# Set up basic logging
logging.basicConfig(level=logging.INFO)

def run_pipeline(file_path):
    """Run the entire pipeline from loading data to saving the model."""
    
    # Step 1: Load the data
    try:
        logging.info("Loading data...")
        data = load_data(file_path)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return
    
    # Step 2: Split the data into training and testing sets
    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Determine which columns are numerical and categorical
    numerical_features = X_train.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Step 3: Preprocess the data
    logging.info("Preprocessing data...")
    X_train_scaled, X_test_scaled, preprocessor = preprocess_data(X_train, X_test, numerical_features, categorical_features)

    # Step 4: Train the model
    logging.info("Training the model...")
    model = train_model(X_train_scaled, y_train)
    
    # Step 5: Evaluate the model
    logging.info("Evaluating the model...")
    mae, mse, r2 = evaluate_model(model, X_test_scaled, y_test)
    logging.info(f"Model performance on test data:\nMAE: {mae:.4f}\nMSE: {mse:.4f}\nR²: {r2:.4f}")
    
    # Step 6: Save the model
    save_model(model, 'model.pkl')
    logging.info("Pipeline completed successfully.")

# if __name__ == "__main__":
#     # Provide the path to your dataset
#     file_path = 'C:\\Users\\DELL\\Downloads\\ModularProjects\\AmesHousing\\data\\AmesHousing.csv'
#     run_pipeline(file_path)
