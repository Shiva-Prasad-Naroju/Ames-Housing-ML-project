import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# finding the missing value columns
def find_missing_value_cols(dataframe):
    nulls = dataframe.isna().sum()
    missing_cols = nulls[nulls>0]
    return missing_cols


# finding the missing value percentage in features:
def find_missing_value_percentage(df):
    for i in [i for i in df.columns if df[i].isna().sum()>0]:
        print(i,' - ',np.round(df[i].isna().mean()*100,2), '% missing values')


# getting the cols considering the threshold more and less Nan values:
def get_nan_cols_with_threshold(df, threshold=20):
    missing_percentages = df.isna().mean() * 100
    high_missing_cols = missing_percentages[missing_percentages > threshold].index.tolist()
    low_missing_cols = missing_percentages[(missing_percentages > 0) & (missing_percentages <= threshold)].index.tolist()
    return high_missing_cols, low_missing_cols



# removing the missing columns which has higher percentage of Nan value
def remove_missing_cols(df,high_missing_cols):
    df.drop(columns=high_missing_cols,inplace=True)
    return df

# deleting the duplicate rows
def delete_duplicate_rows(dataframe):
    dataframe = dataframe.drop_duplicates(keep="first")
    return dataframe





### IDENTIFY AND CATEGORIZE COLUMNS:

# finding the numerical columns
def find_num_cols(df):
    """
    Returns numerical columns in the DataFrame.
    """
    return [i for i in df.columns if df[i].dtype != 'O']


# finding the categorical columns
def find_cat_cols(df):
    """
    Returns categorical columns in the DataFrame.
    """
    return [i for i in df.columns if df[i].dtype == 'O']


# finding the discrete datatype columns
def find_discrete_data_cols(df):
    """
    Returns discrete numerical columns (with < 25 unique values).
    """
    num_cols = find_num_cols(df)
    return [i for i in num_cols if len(df[i].unique()) < 10]


# finding the continuous datatype columns
def find_continuous_data_cols(df):
    """
    Returns continuous numerical columns (with 10+ unique values).
    """
    num_cols = find_num_cols(df)
    discrete_var_cols = find_discrete_data_cols(df)
    return [i for i in num_cols if i not in discrete_var_cols]


# finding the numerical columns which has Nan values
def find_num_cols_with_nan(df):
    """
    Returns numerical columns with missing values (NaNs).
    """
    return [i for i in df.columns if df[i].isna().sum() > 0 and df[i].dtypes != 'O']


# finding the categorical columns which has Nan values
def find_cat_cols_with_nan(df):
    """
    Returns categorical columns with missing values (NaNs).
    """
    return [i for i in df.columns if df[i].isna().sum() > 0 and df[i].dtypes == 'O']



### HANDLE MISSING VALUES:
# =======================

# Finding the numerical columns to impute the Nan values, this is based on the threshold value
def get_num_cols_to_impute_nan(df):
    _,low_missing_cols = get_nan_cols_with_threshold(df)
    num_cols = [i for i in low_missing_cols if df[i].dtype!='O']   # don't use the list name as same function name, we get error while executing.
    return num_cols


# Finding the categorical columns to impute Nan values , this is based on the threshold
def get_cat_cols_to_imute_nan(df):
    _,low_missing_cols = get_nan_cols_with_threshold(df)
    cat_cols = [i for i in low_missing_cols if df[i].dtype=='O']  # don't use the list name as same function name, we get error while executing.
    return cat_cols


# Imputing the numerical columns with median
def impute_num_cols(df,num_cols_to_impute_nan):
    for i in num_cols_to_impute_nan:
        if df[i].isna().sum()>0:
            df[i] = df[i].fillna(df[i].median())
        else:
            df[i] = df[i]
        

# Imputing the categorical columns with mode
def impute_cat_cols(df,cat_cols_to_impute_nan):
    for i in cat_cols_to_impute_nan:
        if df[i].isna().sum()>0:
            df[i] = df[i].fillna(df[i].mode()[0])
        else:
            df[i] = df[i]



### HANDLE CONSTANT FEATURES:
# ==========================

# Find the columns which has only 1 unique value in it (or 1 label) throughout all the records.
def find_constant_columns(dataframe):
    constant_columns = []
    for column in dataframe.columns:
        # Get unique values in the column
        unique_values = dataframe[column].unique()
        # check if the column contains only one unique value
        if len(unique_values) == 1:
            constant_columns.append(column)
    return constant_columns


# Delete the constant columns from the dataframe
def delete_constant_columns(dataframe, columns_to_delete):
    # Delete the specified columns
    dataframe = dataframe.drop(columns_to_delete, axis=1)
    return dataframe


# Find the columns with only few unique values 
def find_columns_with_few_values(dataframe, threshold):
    few_values_columns = []
    for column in dataframe.columns:
        # Get the number of unique values in the column
        unique_values_count = len(dataframe[column].unique())
        # Check if the column has less than the threshold number of unique values
        if unique_values_count < threshold:
            few_values_columns.append(column)
    return few_values_columns






### HANDLE OUTLIERS:
# =================

# finding outlier columns
def find_outlier_cols(df):
    outlier_cols = []
    for i in find_continuous_data_cols(df):
        q1 = df[i].quantile(0.25)  
        q3 = df[i].quantile(0.75)  
        iqr = q3 - q1  

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        if ((df[i] < lower_limit) | (df[i] > upper_limit)).any():
            outlier_cols.append(i)
    return outlier_cols



# Creates a table of columns with their outlier percentages
def get_outlier_percentages(df):
    outlier_cols = find_outlier_cols(df)
    data = []
    for col in outlier_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_percentage = (outliers / len(df)) * 100
        data.append([col, outlier_percentage])
    return pd.DataFrame(data, columns=["Column", "Outlier_Percentage"]).sort_values(by="Outlier_Percentage", ascending=False)



# Remove the columns with outliers considering more than threshold outlier percentage:
def remove_columns_with_high_outliers(df, threshold=5):
    outlier_cols = find_outlier_cols(df)
    cols_to_remove = []
    for col in outlier_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Calculate the percentage of outliers in the column
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_percentage = (outliers / len(df)) * 100

        if outlier_percentage > threshold:
            cols_to_remove.append(col)
    if cols_to_remove:
        print(f"Removed columns due to high outlier percentage (> {threshold}%): {cols_to_remove}")
    else:
        print("No columns were removed due to high outlier percentage.")
    df.drop(columns=cols_to_remove, inplace=True)
    return df



# Categorising the outliers into cols to cap and cols to remove
def categorize_outlier_cols(df, threshold=10):
    cols_to_cap_outliers = []
    cols_to_remove_outliers_records = []
    outlier_cols = find_outlier_cols(df)
    for col in outlier_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_percentage = (outliers / len(df)) * 100
        if outlier_percentage > threshold:
            cols_to_cap_outliers.append(col)
        else:
            cols_to_remove_outliers_records.append(col)
    return cols_to_cap_outliers,cols_to_remove_outliers_records


def cap_outliers(df):
    cols_to_cap,_ = categorize_outlier_cols(df)
    for i in cols_to_cap:
        q1 = df[i].quantile(0.25)  # 25th percentile
        q3 = df[i].quantile(0.75)  # 75th percentile
        iqr = q3 - q1  # Interquartile Range
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        # Cap outliers in the column
        df[i] = df[i].apply(lambda x: lower_limit if x < lower_limit else (upper_limit if x > upper_limit else x))
        return df

def remove_outlier_records(df):
    _, cols_to_remove = categorize_outlier_cols(df)
    for col in cols_to_remove:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Remove rows where the column value is an outlier
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df




#### DATA SPLITTING

# Splitting the data
def split_the_data(df,target_column):
    X = df.drop(target_column,axis=1)
    y = df[target_column]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=43)
    return (X_train,X_test,y_train,y_test)

