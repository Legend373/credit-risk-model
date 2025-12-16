import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =====================================================
# PROXY TARGET VARIABLE ENGINEERING (RFM + CLUSTERING)
# =====================================================

class ProxyTargetEngineer:
    """
    Creates a proxy credit-risk target variable (is_high_risk)
    using RFM analysis and K-Means clustering.
    """

    def __init__(
        self,
        customer_col: str = "CustomerId",
        date_col: str = "TransactionStartTime",
        amount_col: str = "Amount",
        n_clusters: int = 3,
        random_state: int = 42
    ):
        self.customer_col = customer_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.scaler_ = StandardScaler()
        self.kmeans_ = None
        self.rfm_df_ = None
        self.high_risk_cluster_ = None

    # =====================================================
    # 1. COMPUTE RFM METRICS
    # =====================================================
    def compute_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Recency, Frequency, Monetary metrics per customer
        """

        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")

        snapshot_date = df[self.date_col].max() + pd.Timedelta(days=1)

        rfm = (
            df.groupby(self.customer_col)
            .agg(
                Recency=(self.date_col, lambda x: (snapshot_date - x.max()).days),
                Frequency=(self.customer_col, "count"),
                Monetary=(self.amount_col, "sum")
            )
            .reset_index()
        )

        self.rfm_df_ = rfm
        return rfm

    # =====================================================
    # 2. FIT CLUSTERING MODEL
    # =====================================================
    def fit(self, df: pd.DataFrame):
        """
        Fit KMeans clustering on scaled RFM features
        """

        rfm = self.compute_rfm(df)

        rfm_features = rfm[["Recency", "Frequency", "Monetary"]]

        rfm_scaled = self.scaler_.fit_transform(rfm_features)

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        rfm["cluster"] = self.kmeans_.fit_predict(rfm_scaled)

        # Identify high-risk cluster:
        # High Recency (inactive), Low Frequency, Low Monetary
        cluster_profile = (
            rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]]
            .mean()
        )

        self.high_risk_cluster_ = (
            cluster_profile
            .sort_values(by=["Recency", "Frequency", "Monetary"],
                          ascending=[False, True, True])
            .index[0]
        )

        self.rfm_df_ = rfm
        return self

    # =====================================================
    # 3. ASSIGN HIGH-RISK LABEL
    # =====================================================
    def transform(self) -> pd.DataFrame:
        """
        Assign is_high_risk label based on cluster
        """

        if self.rfm_df_ is None or self.high_risk_cluster_ is None:
            raise RuntimeError("You must call .fit() before .transform()")

        rfm_labeled = self.rfm_df_.copy()
        rfm_labeled["is_high_risk"] = (
            rfm_labeled["cluster"] == self.high_risk_cluster_
        ).astype(int)

        return rfm_labeled[
            [self.customer_col, "is_high_risk"]
        ]

    # =====================================================
    # 4. MERGE BACK TO MAIN DATASET
    # =====================================================
    def merge_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge proxy target back to transaction-level dataset
        """

        target_df = self.transform()

        df_final = df.merge(
            target_df,
            on=self.customer_col,
            how="left"
        )

        df_final["is_high_risk"] = df_final["is_high_risk"].fillna(0).astype(int)

        return df_final
