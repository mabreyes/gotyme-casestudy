"""Visualization module for model analysis."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_feature_importance(model, feature_names, save_path=None):
    """Plot feature importance from the XGBoost model.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    importance_scores = model.feature_importances_
    sorted_idx = np.argsort(importance_scores)
    pos = np.arange(sorted_idx.shape[0]) + 0.5

    plt.barh(pos, importance_scores[sorted_idx])
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel("Feature Importance Score")
    plt.title("Feature Importance")

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot ROC curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Optional path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_financial_impact(analysis_results, save_path=None):
    """Plot financial impact analysis results.

    Args:
        analysis_results: Dictionary containing financial analysis results
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))

    # Plot profit by risk band
    risk_bands = list(analysis_results["profit_by_risk_band"].keys())
    profits = list(analysis_results["profit_by_risk_band"].values())

    plt.subplot(1, 2, 1)
    colors = ["red", "yellow", "green"]
    plt.bar(risk_bands, profits, color=colors)
    plt.title("Profit by Risk Band")
    plt.xlabel("Risk Band")
    plt.ylabel("Profit")

    # Plot total metrics
    plt.subplot(1, 2, 2)
    metrics = ["Total Profit", "Opportunity Loss"]
    values = [analysis_results["total_profit"], analysis_results["opportunity_loss"]]
    plt.bar(metrics, values, color=["blue", "orange"])
    plt.title("Overall Financial Metrics")
    plt.ylabel("Amount")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()
