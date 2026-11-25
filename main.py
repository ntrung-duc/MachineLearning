import time
from pathlib import Path

import joblib

from src.evaluation import (
    METRICS_PATH,
    evaluate_model,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_roc_curves,
    save_metrics,
    select_best_model,
)
from src.naive_bayes import train_naive_bayes, tune_naive_bayes
from src.preprocessing import prepare_data
from src.random_forest import train_random_forest, tune_random_forest
from src.svm_model import train_svm, tune_svm

MODELS_DIR = Path(__file__).resolve().parent / "models"

STEPS = [
    ("Naive Bayes", "nb", train_naive_bayes, tune_naive_bayes),
    ("SVM", "svm", train_svm, tune_svm),
    ("Random Forest", "rf", train_random_forest, tune_random_forest),
]


def run():
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, Cancel rate train: {y_train.mean():.4f}")

    if METRICS_PATH.exists():
        METRICS_PATH.unlink()

    rows = []
    tuned_models = {}
    for name, short, train_fn, tune_fn in STEPS:
        start = time.time()
        baseline = train_fn(X_train, y_train)
        rows.append(evaluate_model(baseline, X_test, y_test, name, "baseline"))
        tuned, params, cv_score = tune_fn(X_train, y_train)
        rows.append(evaluate_model(tuned, X_test, y_test, name, "tuned"))
        tuned_models[name] = tuned
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(tuned, MODELS_DIR / f"{short}_tuned.joblib")
        plot_confusion_matrix(tuned, X_test, y_test, f"{name} (tuned)", f"confusion_matrix_{short}.png")
        print(f"{name}: best params {params}, CV F1 {cv_score:.4f}, time {time.time() - start:.1f}s")

    metrics = save_metrics(rows)
    plot_metric_comparison(metrics, "model_comparison.png")
    plot_roc_curves(tuned_models, X_test, y_test, "roc_curves.png")
    print(metrics.to_string(index=False))

    best = select_best_model(metrics[metrics["stage"] == "tuned"])
    joblib.dump(tuned_models[best["model"]], MODELS_DIR / "best_model.joblib")
    print(f"Best model: {best['model']} (F1 {best['f1']:.4f}, ROC-AUC {best['roc_auc']:.4f})")


if __name__ == "__main__":
    run()
