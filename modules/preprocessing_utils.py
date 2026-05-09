"""
Preprocessing utilities for the Churn Prediction Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA


def get_feature_types(df, target_col='Churned'):
    """
    Separate features into categorical and numerical types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column to exclude
    
    Returns:
    --------
    dict with 'categorical' and 'numerical' lists
    """
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if target_col in numerical_features:
        numerical_features.remove(target_col)
    
    return {
        'categorical': categorical_features,
        'numerical': numerical_features
    }


def build_preprocessing_pipeline(categorical_features, numerical_features, 
                                 scaler_type='standard', scaler_params=None):
    """
    Build a preprocessing pipeline with imputation, encoding, and scaling.
    
    Parameters:
    -----------
    categorical_features : list
        List of categorical feature names
    numerical_features : list
        List of numerical feature names
    scaler_type : str
        Type of scaler: 'standard', 'minmax', or 'none'
    scaler_params : dict
        Additional parameters for scaler (e.g., feature_range for MinMaxScaler)
    
    Returns:
    --------
    sklearn.compose.ColumnTransformer
        Preprocessing pipeline
    """
    
    # Numerical preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', _get_scaler(scaler_type, scaler_params))
    ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor


def _get_scaler(scaler_type, params=None):
    """Get scaler object based on type."""
    if scaler_type == 'standard':
        return StandardScaler()
    elif scaler_type == 'minmax':
        if params and 'feature_range' in params:
            return MinMaxScaler(feature_range=params['feature_range'])
        else:
            return MinMaxScaler()
    else:
        return 'passthrough'


def analyze_missing_values(df, target_col='Churned'):
    """
    Analyze missing value patterns in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
    
    Returns:
    --------
    pd.DataFrame with missing value statistics
    """
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )
    
    return missing_summary


def apply_pca(X_train, X_test, variance_threshold=0.95):
    """
    Apply PCA to training and test data.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    variance_threshold : float
        Variance threshold for PCA (0.95 = 95%)
    
    Returns:
    --------
    dict with PCA model and transformed data
    """
    pca = PCA(n_components=variance_threshold)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    return {
        'model': pca,
        'X_train_pca': X_train_pca,
        'X_test_pca': X_test_pca,
        'n_components': pca.n_components_,
        'explained_variance': pca.explained_variance_ratio_.sum()
    }


def get_class_weight(y):
    """
    Calculate class weights for imbalanced classification.
    
    Parameters:
    -----------
    y : array-like
        Target vector
    
    Returns:
    --------
    dict with class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return {classes[i]: weights[i] for i in range(len(classes))}
