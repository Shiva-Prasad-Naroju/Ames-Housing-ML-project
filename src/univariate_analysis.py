import numpy as np
import pandas as pd
import nbformat
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objects as go
from scipy.stats import skew, kurtosis


#### Plots:

# Plots the distribution of a numerical feature.
def plot_numerical_distribution(df, num_col):
    plt.figure(figsize=(5, 4))
    sns.histplot(df[num_col], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {num_col}')
    plt.xlabel(num_col)
    plt.ylabel('Frequency')
    plt.show()

### Outlier analysis:
# ==================

# Visualising the outlier in the target column:
def plot_boxplot_for_target_col(df, target_col):
    plt.figure(figsize=(5, 4))
    fig = go.Figure(data=[go.Box(y=df[target_col], boxpoints='outliers', jitter=0.3, pointpos=-1.8)])
    fig.update_layout(
        title=f'Boxplot of {target_col}',
        yaxis_title=f'{target_col}',
        width=700,
        height=500)
    fig.show()
    plt.show



# Plots the outlier in the box plots:
def plot_boxplots(df, continuous_cols):
    for i in continuous_cols:
        plt.figure(figsize=(5, 4))
        fig = go.Figure(data=[go.Box(y=df[i], boxpoints='outliers', jitter=0.3, pointpos=-1.8)])
        fig.update_layout(
            title=f'Boxplot of {i}',
            yaxis_title=f'{i}',
            width=700,
            height=500)
        fig.show()
        plt.show



#### Skewness analysis:

#  Calculates skewness and kurtosis for a numerical feature.
def calculate_skewness_and_kurtosis(df, num_col):
    skewness = skew(df[num_col].dropna())
    kurt = kurtosis(df[num_col].dropna())
    return {'Skewness': skewness, 'Kurtosis': kurt}

