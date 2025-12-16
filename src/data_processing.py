import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# =====================================================
# 1. CONSTANTS
# =====================================================

TARGET_COL = "FraudResult"
CUSTOMER_ID_COL = "CustomerId"
AMOUNT_COL = "Amount"
DATE_COL = "TransactionStartTime"

DROP_COLUMNS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "ProductId",
    CUSTOMER_ID_COL,
    AMOUNT_COL,
    DATE_COL
]

NUMERICAL_FEATURES = ["Value"]

CATEGORICAL_OHE = [
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "PricingStrategy"
]

CATEGORICAL_WOE = [
    "ProductCategory",
    "ChannelId"
]

AGG_FEATURES = [
    "total_transaction_amount",
    "average_transaction_amount",
    "transaction_count",
    "std_transaction_amount"
]

DATE_FEATURES = [
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_year"
]

# =====================================================
# 2. WoE / IV
# =====================================================

def compute_woe_iv(feature, target):
    df = pd.DataFrame({"x": feature.astype(str), "y": target})
    eps = 1e-6

    total_good = (df.y == 0).sum()
    total_bad = (df.y == 1).sum()

    woe_map, iv = {}, 0.0

    for val, g in df.groupby("x"):
        good = (g.y == 0).sum()
        bad = (g.y == 1).sum()

        gr = (good + eps) / (total_good + eps)
        br = (bad + eps) / (total_bad + eps)

        woe = np.log(gr / br)
        woe_map[val] = woe
        iv += (gr - br) * woe

    return woe_map, iv

# =====================================================
# 3. AGGREGATION
# =====================================================

class AggregationFeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.stats_ = (
            X.groupby(CUSTOMER_ID_COL)[AMOUNT_COL]
            .agg(
                total_transaction_amount="sum",
                average_transaction_amount="mean",
                transaction_count="count",
                std_transaction_amount="std"
            )
            .fillna(0)
        )
        return self

    def transform(self, X):
        return X.merge(self.stats_, on=CUSTOMER_ID_COL, how="left")

# =====================================================
# 4. DATE FEATURES (NO SCALING)
# =====================================================

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        dt = pd.to_datetime(X[DATE_COL], errors="coerce")

        X["transaction_hour"] = dt.dt.hour
        X["transaction_day"] = dt.dt.day
        X["transaction_month"] = dt.dt.month
        X["transaction_year"] = dt.dt.year

        return X

# =====================================================
# 5. MAIN PROCESSOR (FIXED)
# =====================================================

class ModelReadyDataProcessor:
    """
    Produces a MODEL-CORRECT dataset:
    - Date features preserved as integers
    - Numeric & aggregate features scaled
    - IDs removed
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.woe_maps_ = {}
        self.iv_summary_ = None
        self.final_columns_ = None

    # =============================
    # FIT
    # =============================
    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = y.astype(int)

        X = AggregationFeatureGenerator().fit_transform(X)
        X = DateFeatureExtractor().transform(X)

        # ---- WoE ----
        iv_rows = []
        for col in CATEGORICAL_WOE:
            woe, iv = compute_woe_iv(X[col], y)
            self.woe_maps_[col] = woe
            iv_rows.append({"Feature": col, "IV": iv})

        self.iv_summary_ = pd.DataFrame(iv_rows).sort_values("IV", ascending=False)

        # ---- OHE ----
        self.ohe.fit(X[CATEGORICAL_OHE])

        # ---- Scale ONLY numeric + aggregate ----
        self.scaler.fit(X[NUMERICAL_FEATURES + AGG_FEATURES])

        self.final_columns_ = (
            NUMERICAL_FEATURES +
            AGG_FEATURES +
            DATE_FEATURES +
            list(self.ohe.get_feature_names_out(CATEGORICAL_OHE)) +
            [f"{c}_woe" for c in CATEGORICAL_WOE]
        )

        return self

    # =============================
    # TRANSFORM
    # =============================
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = AggregationFeatureGenerator().fit_transform(X)
        X = DateFeatureExtractor().transform(X)

        # ---- Scale numeric + aggregate ONLY ----
        scaled = self.scaler.transform(X[NUMERICAL_FEATURES + AGG_FEATURES])
        scaled_df = pd.DataFrame(
            scaled,
            columns=NUMERICAL_FEATURES + AGG_FEATURES,
            index=X.index
        )

        date_df = X[DATE_FEATURES].astype("Int64")

        # ---- OHE ----
        ohe_df = pd.DataFrame(
            self.ohe.transform(X[CATEGORICAL_OHE]),
            columns=self.ohe.get_feature_names_out(CATEGORICAL_OHE),
            index=X.index
        )

        # ---- WoE ----
        woe_df = pd.DataFrame({
            f"{c}_woe": X[c].astype(str).map(self.woe_maps_[c]).fillna(0)
            for c in CATEGORICAL_WOE
        }, index=X.index)

        X_final = pd.concat(
            [scaled_df, date_df, ohe_df, woe_df],
            axis=1
        )

        return X_final[self.final_columns_]

    # =============================
    # SAVE
    # =============================
    def save_processed(self, X, path="../data/processed/processed.csv"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        X.to_csv(path, index=False)
        print(f"✅ Saved to {path.resolve()}")
