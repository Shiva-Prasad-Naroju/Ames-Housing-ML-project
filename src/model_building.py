import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Finding the correlation among the numeric features
def correlation_among_numeric_features(df, cols):
    numeric_col = df[cols]
    corr = numeric_col.corr()
    # get highly correlated features and also tell to which feature it is correlated
    corr_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                colname = corr.columns[i]
                corr_features.add(colname)
    return corr_features


# building the Random Forest Regressor Model
def apply_random_forest_regressor(X_train, y_train, n_estimators=100, random_state=42):

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

# evaluate metrics
def evaluate_model_metrics(model, X_test, y_test):
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    r2 = r2_score(y_test, y_pred)  # R-squared value
    
    # Adjusted R-squared
    n = len(y_test)  # Number of data points
    p = X_test.shape[1]  # Number of features
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Prepare a dictionary to store all metrics
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Adjusted R2': adj_r2,
        'MAPE': mape
    }
    
    # Print all metrics
    for metric, value in metrics.items():
        print(f'{metric}: {np.round(value,2)}')
    
    return metrics


# Hyperparameter tuning using GridSearchCV for RandomForestRegressor
def tune_random_forest_hyperparameters(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for the best split
        'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)
    # Get the best model and hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f'Best Hyperparameters: {best_params}')
    return best_model, best_params


# building the Random Forest Regressor Model using the best hyperparameters, # use this if performed hyptuning:
def apply_random_forest_with_best_params(X_train_scaled_1, y_train, best_params):
    """
    Builds and fits the Random Forest Regressor model using the best hyperparameters from GridSearchCV.
    Parameters:
    - X_train: Training feature set
    - y_train: Target variable for training
    - best_params: Dictionary containing the best hyperparameters from GridSearchCV
    Returns:
    - model: Trained RandomForestRegressor model
    """
    # Initialize the RandomForestRegressor model using the best parameters
    model1 = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        bootstrap=best_params['bootstrap'],
        random_state=42
    )
    # Fit the model on the training data
    model1.fit(X_train_scaled_1, y_train)
    
    return model1




