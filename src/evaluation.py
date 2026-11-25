from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_PATH = RESULTS_DIR / "metrics.csv"
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
CLASS_NAMES = ["Not canceled", "Canceled"]


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def evaluate_model(model, X_test, y_test, name, stage):
    y_pred = model.predict(X_test)
    y_score = get_scores(model, X_test)
    return {
        "model": name,
        "stage": stage,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


def print_report(model, X_test, y_test):
    print(classification_report(y_test, model.predict(X_test), target_names=CLASS_NAMES, digits=4))


def plot_confusion_matrix(model, X_test, y_test, title, filename=None, ax=None):
    matrix = confusion_matrix(y_test, model.predict(X_test))
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    if filename:
        save_figure(ax.figure, filename)
    return ax


def plot_roc_curves(models, X_test, y_test, filename=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        RocCurveDisplay.from_predictions(y_test, get_scores(model, X_test), name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC curves on test set")
    if filename:
        save_figure(fig, filename)
    return ax


def plot_metric_comparison(metrics, filename=None):
    data = metrics.melt(id_vars=["model", "stage"], value_vars=METRIC_COLUMNS, var_name="metric", value_name="score")
    data["label"] = data["model"] + " (" + data["stage"] + ")"
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=data, x="metric", y="score", hue="label", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on test set")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    if filename:
        save_figure(fig, filename)
    return ax


def save_figure(fig, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=120, bbox_inches="tight")


def save_metrics(rows, path=METRICS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        keys = set(zip(table["model"], table["stage"]))
        old = old[[pair not in keys for pair in zip(old["model"], old["stage"])]]
        table = pd.concat([old, table], ignore_index=True)
    table[METRIC_COLUMNS] = table[METRIC_COLUMNS].round(4)
    table.to_csv(path, index=False)
    return table


def load_metrics(path=METRICS_PATH):
    return pd.read_csv(path)


def select_best_model(metrics, metric="f1"):
    return metrics.sort_values([metric, "roc_auc"], ascending=False).iloc[0]
