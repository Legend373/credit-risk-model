import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# =====================================================
# 1. COLUMN DEFINITIONS
# =====================================================

TARGET_COL = "FraudResult"
CUSTOMER_ID_COL = "CustomerId"
AMOUNT_COL = "Amount"
DATE_COL = "TransactionStartTime"

NUMERICAL_FEATURES = ["Value"]

CATEGORICAL_FEATURES_OHE = [
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "PricingStrategy"
]

CATEGORICAL_FEATURES_WOE = [
    "ProductCategory",
    "ChannelId"
]

# =====================================================
# 2. WoE & IV
# =====================================================

def compute_woe_iv(feature, target):
    df = pd.DataFrame({
        "feature": feature.astype(str),
        "target": target
    })

    total_good = (df["target"] == 0).sum()
    total_bad = (df["target"] == 1).sum()
    eps = 1e-6

    woe_map = {}
    iv = 0.0

    for cat, group in df.groupby("feature"):
        good = (group["target"] == 0).sum()
        bad = (group["target"] == 1).sum()

        good_rate = (good + eps) / (total_good + eps)
        bad_rate = (bad + eps) / (total_bad + eps)

        woe = np.log(good_rate / bad_rate)
        woe_map[cat] = woe
        iv += (good_rate - bad_rate) * woe

    return woe_map, iv

# =====================================================
# 3. AGGREGATION FEATURES
# =====================================================

class AggregationFeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.agg_ = (
            X.groupby(CUSTOMER_ID_COL)[AMOUNT_COL]
            .agg(
                total_transaction_amount="sum",
                average_transaction_amount="mean",
                transaction_count="count",
                std_transaction_amount="std"
            )
            .fillna(0)
            .reset_index()
        )
        return self

    def transform(self, X):
        return X.merge(self.agg_, on=CUSTOMER_ID_COL, how="left")

# =====================================================
# 4. DATE FEATURES
# =====================================================

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[DATE_COL] = pd.to_datetime(X[DATE_COL], errors="coerce")

        X["transaction_hour"] = X[DATE_COL].dt.hour
        X["transaction_day"] = X[DATE_COL].dt.day
        X["transaction_month"] = X[DATE_COL].dt.month
        X["transaction_year"] = X[DATE_COL].dt.year

        return X.drop(columns=[DATE_COL])

# =====================================================
# 5. MAIN PROCESSOR (WITH SAVE)
# =====================================================

class ModelReadyDataProcessor:
    """
    Full feature engineering pipeline + WoE + IV
    """

    def __init__(self, scaling_method="Standard"):
        self.scaler = (
            StandardScaler() if scaling_method == "Standard"
            else MinMaxScaler()
        )
        self.woe_maps_ = {}
        self.iv_summary_ = None
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", self.scaler)
        ])

        ohe_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, NUMERICAL_FEATURES),
                ("ohe", ohe_pipeline, CATEGORICAL_FEATURES_OHE),
                ("woe_imp",
                 SimpleImputer(strategy="constant", fill_value="missing"),
                 CATEGORICAL_FEATURES_WOE),
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

        return Pipeline([
            ("aggregation", AggregationFeatureGenerator()),
            ("date_features", DateFeatureExtractor()),
            ("preprocessing", preprocessor)
        ])

    # =============================
    # FIT
    # =============================
    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = y.astype(int)

        X_processed = self.pipeline.fit_transform(X, y)

        feature_names = (
            self.pipeline.named_steps["preprocessing"]
            .get_feature_names_out()
        )

        X_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

        iv_records = []
        for col in CATEGORICAL_FEATURES_WOE:
            woe_map, iv = compute_woe_iv(X_df[col], y)
            self.woe_maps_[col] = woe_map
            iv_records.append({"Feature": col, "IV": iv})

        self.iv_summary_ = (
            pd.DataFrame(iv_records)
            .sort_values("IV", ascending=False)
            .reset_index(drop=True)
        )

        return self

    # =============================
    # TRANSFORM
    # =============================
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = self.pipeline.transform(X)

        feature_names = (
            self.pipeline.named_steps["preprocessing"]
            .get_feature_names_out()
        )

        X_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

        woe_features = {}
        for col in CATEGORICAL_FEATURES_WOE:
            woe_features[f"{col}_woe"] = (
                X_df[col].astype(str)
                .map(self.woe_maps_[col])
                .fillna(0.0)
            )

        X_woe = pd.DataFrame(woe_features, index=X.index)

        X_final = pd.concat(
            [X_df.drop(columns=CATEGORICAL_FEATURES_WOE), X_woe],
            axis=1
        )

        return X_final

    # =============================
    # SAVE TO CSV
    # =============================
    def save_processed(self, X: pd.DataFrame,path: str = "../data/processed/processed.csv"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        X.to_csv(path, index=False)
        print(f"✅ Processed data saved to: {path.resolve()}")
