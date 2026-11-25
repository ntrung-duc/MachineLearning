from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "dataset" / "hotel_bookings.csv"
TARGET = "is_canceled"
TEST_SIZE = 0.2
RANDOM_STATE = 42

LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date", "assigned_room_type"]
DROP_COLUMNS = ["company", "agent", "arrival_date_year", "arrival_date_week_number"]

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}

CATEGORICAL_FEATURES = [
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

NUMERIC_FEATURES = [
    "lead_time",
    "arrival_month",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "total_nights",
    "total_guests",
    "has_agent",
    "has_company",
    "is_family",
    "has_previous_cancel",
    "month_sin",
    "month_cos",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def data_quality_report(df):
    missing = df.isna().sum()
    report = pd.DataFrame({
        "missing": missing,
        "missing_pct": (missing / len(df) * 100).round(2),
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
    })
    return report.sort_values("missing", ascending=False)


def clean_data(df):
    data = df.copy()
    data = data.drop_duplicates()
    data = data[data["adr"] >= 0]
    data = data[data["adr"] < 1000]
    guests = data["adults"] + data["children"].fillna(0) + data["babies"]
    data = data[guests > 0]
    nights = data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
    data = data[nights > 0]
    data = data[data["adults"] <= 10]
    data["meal"] = data["meal"].replace("Undefined", "SC")
    data["market_segment"] = data["market_segment"].replace("Undefined", np.nan)
    data["distribution_channel"] = data["distribution_channel"].replace("Undefined", np.nan)
    return data.reset_index(drop=True)


def engineer_features(df):
    data = df.copy()
    data["arrival_month"] = data["arrival_date_month"].map(MONTHS)
    data["month_sin"] = np.sin(2 * np.pi * data["arrival_month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["arrival_month"] / 12)
    data["total_nights"] = data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
    data["total_guests"] = data["adults"] + data["children"].fillna(0) + data["babies"]
    data["has_agent"] = data["agent"].notna().astype(int)
    data["has_company"] = data["company"].notna().astype(int)
    data["is_family"] = ((data["children"].fillna(0) + data["babies"]) > 0).astype(int)
    data["has_previous_cancel"] = (data["previous_cancellations"] > 0).astype(int)
    data = data.drop(columns=["arrival_date_month"] + LEAKAGE_COLUMNS + DROP_COLUMNS)
    return data


def prepare_data(path=DATA_PATH):
    df = engineer_features(clean_data(load_data(path)))
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def build_preprocessor(scale=False, dense=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200, sparse_output=not dense)),
    ]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", Pipeline(categorical_steps), CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0 if dense else 0.3,
    )


def get_feature_names(preprocessor):
    return [name.split("__", 1)[1] for name in preprocessor.get_feature_names_out()]
