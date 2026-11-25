import sys
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import FEATURES, TARGET, clean_data, engineer_features, load_data

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model.joblib"


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError("Model not found. Run: python main.py")
    return joblib.load(path)


def predict_bookings(raw_bookings, model=None):
    model = model or load_model()
    X = engineer_features(raw_bookings)[FEATURES]
    result = pd.DataFrame({"prediction": model.predict(X)}, index=raw_bookings.index)
    if hasattr(model, "predict_proba"):
        try:
            result["cancel_probability"] = model.predict_proba(X)[:, 1].round(4)
        except AttributeError:
            pass
    return result


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sample = clean_data(load_data()).sample(n, random_state=7)
    output = predict_bookings(sample)
    output.insert(0, "actual", sample[TARGET].values)
    print(output.to_string())
