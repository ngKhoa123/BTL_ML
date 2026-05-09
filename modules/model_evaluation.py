"""
Model evaluation utilities for the Churn Prediction Pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)


def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Compute comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for positive class
    
    Returns:
    --------
    dict with all computed metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def print_classification_report(y_true, y_pred, model_name='Model'):
    """
    Print detailed classification report.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
    """
    print(f"\n{'='*70}")
    print(f"Classification Report: {model_name}")
    print(f"{'='*70}")
    print(classification_report(y_true, y_pred, target_names=['Not Churned', 'Churned']))


def plot_confusion_matrix(y_true, y_pred, model_name='Model', ax=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Not Churned', 'Churned'])
    ax.set_yticklabels(['Not Churned', 'Churned'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    
    return ax


def plot_roc_curve(y_true, y_pred_proba, model_name='Model', ax=None):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve: {model_name}', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return ax


def compare_models_summary(results_dict):
    """
    Create a comparison summary of multiple models.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and results dict as values
    
    Returns:
    --------
    pd.DataFrame with comparison results
    """
    comparison_data = []
    
    for model_name, results in results_dict.items():
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Train Accuracy': results.get('train_accuracy', np.nan),
            'Test Accuracy': results.get('test_accuracy', np.nan),
            'Train Recall': results.get('train_recall', np.nan),
            'Test Recall': results.get('test_recall', np.nan),
            'Train Precision': results.get('train_precision', np.nan),
            'Test Precision': results.get('test_precision', np.nan),
            'Train F1': results.get('train_f1', np.nan),
            'Test F1': results.get('test_f1', np.nan),
            'ROC-AUC': results.get('test_roc_auc', np.nan)
        })
    
    return pd.DataFrame(comparison_data)


def save_model_results(results_dict, filepath):
    """
    Save model results to CSV.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names and results
    filepath : str
        Path to save CSV file
    """
    comparison_df = compare_models_summary(results_dict)
    comparison_df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
