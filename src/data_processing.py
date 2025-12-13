"""
data_processing.py
--------------------------------
Robust, automated, and reproducible data processing pipeline.

✔ Uses sklearn.pipeline.Pipeline end-to-end
✔ Transforms raw transaction data into model-ready format
✔ Saves processed data to: data/processed/proccessed.csv

Features implemented:
- Customer-level aggregate features
- Datetime feature extraction
- Missing value handling
- Categorical encoding (One-Hot or WoE)
- Numerical standardization / normalization
- Optional WoE + IV report generation
"""

import os
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Optional WoE
try:
    from xverse.transformers import WoEEncoder
    WOE_AVAILABLE = True
except ImportError:
    WOE_AVAILABLE = False


# ==================================================
# Custom Transformers
# ==================================================

class TransactionAggregates(BaseEstimator, TransformerMixin):
    """Create customer-level aggregate transaction features."""
    def __init__(self, customer_col="CustomerId", amount_col="Amount"):
        self.customer_col = customer_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        agg = (
            X.groupby(self.customer_col)[self.amount_col]
            .agg(
                TotalTransactionAmount="sum",
                AvgTransactionAmount="mean",
                TransactionCount="count",
                StdTransactionAmount="std",
            )
            .fillna(0)
            .reset_index()
        )
        return X.merge(agg, on=self.customer_col, how="left")


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """Extract hour, day, month, year from TransactionStartTime."""
    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors="coerce")
        X["TransactionHour"] = X[self.datetime_col].dt.hour
        X["TransactionDay"] = X[self.datetime_col].dt.day
        X["TransactionMonth"] = X[self.datetime_col].dt.month
        X["TransactionYear"] = X[self.datetime_col].dt.year
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop identifier and leakage-prone columns."""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")


# ==================================================
# Pipeline Builder
# ==================================================

def build_pipeline(use_woe=False, scale_method="standard"):
    """
    Build full preprocessing pipeline.

    scale_method: "standard" | "minmax"
    use_woe: apply WoE encoding (requires xverse)
    """

    categorical_cols = [
        "CurrencyCode", "ProviderId", "ProductId",
        "ProductCategory", "ChannelId", "PricingStrategy"
    ]

    numerical_cols = [
        "CountryCode", "Amount", "Value",
        "TransactionHour", "TransactionDay",
        "TransactionMonth", "TransactionYear",
        "TotalTransactionAmount", "AvgTransactionAmount",
        "TransactionCount", "StdTransactionAmount"
    ]

    scaler = StandardScaler() if scale_method == "standard" else MinMaxScaler()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])

    if use_woe and WOE_AVAILABLE:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("woe", WoEEncoder(return_df=True)),
        ])
    else:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols),
    ])

    full_pipeline = Pipeline([
        ("datetime", DatetimeFeatures()),
        ("aggregates", TransactionAggregates()),
        ("drop_ids", DropColumns([
            "TransactionId", "BatchId", "AccountId",
            "SubscriptionId", "CustomerId", "TransactionStartTime"
        ])),
        ("preprocess", preprocessor),
    ])

    return full_pipeline


# ==================================================
# Runner
# ==================================================

def run_data_processing(
    raw_path="data/raw/data.csv",
    output_path="data/processed/proccessed.csv",
    target="FraudResult",
    use_woe=False,
    scale_method="standard",
):
    """Run full preprocessing and save model-ready data."""

    df = pd.read_csv(raw_path)
    y = df[target]
    X = df.drop(columns=[target])

    pipeline = build_pipeline(use_woe=use_woe, scale_method=scale_method)
    X_processed = pipeline.fit_transform(X, y)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed_df = pd.DataFrame(X_processed)
    processed_df[target] = y.values
    processed_df.to_csv(output_path, index=False)

    # Export IV report if WoE used
    if use_woe and WOE_AVAILABLE:
        encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["woe"]
        encoder.iv_df_.to_csv("data/processed/iv_report.csv", index=False)

    return processed_df


if __name__ == "__main__":
    run_data_processing()
