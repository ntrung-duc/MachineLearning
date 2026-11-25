import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.preprocessing import RANDOM_STATE, build_preprocessor

PARAM_GRID = {"model__var_smoothing": np.logspace(-11, -1, 11)}


def build_naive_bayes(var_smoothing=1e-9):
    return Pipeline([
        ("preprocessor", build_preprocessor(scale=True, dense=True)),
        ("model", GaussianNB(var_smoothing=var_smoothing)),
    ])


def train_naive_bayes(X_train, y_train):
    model = build_naive_bayes()
    model.fit(X_train, y_train)
    return model


def tune_naive_bayes(X_train, y_train, cv=5, scoring="f1"):
    search = GridSearchCV(
        build_naive_bayes(),
        PARAM_GRID,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_
