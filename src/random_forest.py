import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing import RANDOM_STATE, build_preprocessor, get_feature_names

PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 20, 30, 40],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 0.3],
}


def build_random_forest(**params):
    return Pipeline([
        ("preprocessor", build_preprocessor(scale=False)),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)),
    ])


def train_random_forest(X_train, y_train):
    model = build_random_forest()
    model.fit(X_train, y_train)
    return model


def tune_random_forest(X_train, y_train, n_iter=12, cv=3, scoring="f1"):
    search = RandomizedSearchCV(
        build_random_forest(),
        PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def feature_importance(model, top_n=20):
    names = get_feature_names(model.named_steps["preprocessor"])
    importances = model.named_steps["model"].feature_importances_
    table = pd.DataFrame({"feature": names, "importance": importances})
    return table.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
