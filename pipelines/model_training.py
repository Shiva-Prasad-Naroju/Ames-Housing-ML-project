
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_model(X_train, y_train, model_type='xgboost'):
    """Train a model on the given training data."""
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gradient_descent':
        model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    # Make predictions using the trained model
    predictions = model.predict(X_test)
    
    # Calculate and log evaluation metrics
    mae = mean_absolute_error(y_test,predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return mae, mse, r2
