from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.preprocessing import RANDOM_STATE, build_preprocessor

TRAIN_SAMPLE_SIZE = 20000
TUNE_SAMPLE_SIZE = 6000

PARAM_GRID = [
    {"model__kernel": ["rbf"], "model__C": [0.5, 1, 5, 10], "model__gamma": ["scale", 0.01, 0.05]},
    {"model__kernel": ["linear"], "model__C": [0.1, 1]},
]


def sample_train(X_train, y_train, size):
    if size >= len(X_train):
        return X_train, y_train
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train, train_size=size, random_state=RANDOM_STATE, stratify=y_train
    )
    return X_sample, y_sample


def build_svm(C=1.0, kernel="rbf", gamma="scale"):
    return Pipeline([
        ("preprocessor", build_preprocessor(scale=True)),
        ("model", SVC(C=C, kernel=kernel, gamma=gamma, cache_size=1000, random_state=RANDOM_STATE)),
    ])


def train_svm(X_train, y_train, sample_size=TRAIN_SAMPLE_SIZE):
    X_sample, y_sample = sample_train(X_train, y_train, sample_size)
    model = build_svm()
    model.fit(X_sample, y_sample)
    return model


def tune_svm(X_train, y_train, tune_size=TUNE_SAMPLE_SIZE, sample_size=TRAIN_SAMPLE_SIZE, cv=3, scoring="f1"):
    X_tune, y_tune = sample_train(X_train, y_train, tune_size)
    search = GridSearchCV(
        build_svm(),
        PARAM_GRID,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    search.fit(X_tune, y_tune)
    best = {key.replace("model__", ""): value for key, value in search.best_params_.items()}
    X_sample, y_sample = sample_train(X_train, y_train, sample_size)
    model = build_svm(**best)
    model.fit(X_sample, y_sample)
    return model, search.best_params_, search.best_score_
