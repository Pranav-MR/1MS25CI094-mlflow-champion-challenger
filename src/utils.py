"""
Utility functions for MLOps Champion-Challenger pipeline:
- Environment and configuration helpers
- Evaluation metric calculations
- Visualization generation (confusion matrix, comparison charts)
- Artifact export helpers
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Suppress environment noise
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# SQLite tracking URI for MLflow experiment tracking and registry
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_STUDENT_USN = "1MS25CI094"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"


def get_student_usn() -> str:
    """Return student USN from environment or default."""
    return os.environ.get("STUDENT_USN", DEFAULT_STUDENT_USN).strip()


def get_tracking_uri() -> str:
    """Return MLflow tracking URI."""
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def compute_classification_metrics(y_true, y_pred, latency_ms_list=None, total_time_s=None) -> dict:
    """
    Compute standard classification and latency metrics:
    - Accuracy
    - Precision (macro)
    - Recall (macro)
    - Macro F1-score
    - Average latency (ms)
    - Total inference time (s)
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "macro_f1": round(f1, 4),
    }

    if latency_ms_list is not None and len(latency_ms_list) > 0:
        metrics["avg_latency_ms"] = round(float(np.mean(latency_ms_list)), 2)
    else:
        metrics["avg_latency_ms"] = 0.0

    if total_time_s is not None:
        metrics["total_inference_s"] = round(float(total_time_s), 2)
    else:
        metrics["total_inference_s"] = 0.0

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, output_path: str, title: str = "Confusion Matrix"):
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )
            
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics_a: dict, metrics_b: dict, output_path: str):
    """Generate and save a side-by-side metric comparison chart."""
    metric_keys = ["accuracy", "precision", "recall", "macro_f1"]
    labels = ["Accuracy", "Precision", "Recall", "Macro F1"]
    vals_a = [metrics_a.get(k, 0.0) for k in metric_keys]
    vals_b = [metrics_b.get(k, 0.0) for k in metric_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, vals_a, width, label="Model A (DistilBERT - Champion)", color="#3498db")
    rects2 = ax.bar(x + width/2, vals_b, width, label="Model B (BERT - Challenger)", color="#e67e22")

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Champion vs. Challenger Metric Comparison (SST-2)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metrics_summary(summary_data: dict, output_path: str):
    """Save metrics comparison summary dictionary to a formatted JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
