# tests/test_data_processing.py
import pytest
import pandas as pd
import numpy as np
from src.data_processing import ModelReadyDataProcessor, AggregationFeatureGenerator, DateFeatureExtractor

# =====================================================
# 1️⃣ Sample Data for Testing
# =====================================================
@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        "CustomerId": ["C1", "C2", "C1", "C3"],
        "Amount": [100, 200, 150, 300],
        "TransactionStartTime": pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-02 11:30:00",
            "2024-01-05 14:45:00",
            "2024-01-07 09:15:00"
        ]),
        "Value": [1.0, 2.0, 1.5, 3.0],
        "CurrencyCode": ["USD", "USD", "EUR", "EUR"],
        "CountryCode": ["US", "US", "FR", "FR"],
        "ProviderId": ["P1", "P2", "P1", "P3"],
        "PricingStrategy": ["A", "B", "A", "C"],
        "ProductCategory": ["X", "Y", "X", "Z"],
        "ChannelId": ["CH1", "CH2", "CH1", "CH3"]
    })
    return df

# =====================================================
# 2️⃣ Test AggregationFeatureGenerator
# =====================================================
def test_aggregation_feature_generator(sample_data):
    agg_gen = AggregationFeatureGenerator()
    agg_gen.fit(sample_data)
    transformed = agg_gen.transform(sample_data)

    # Check that new aggregation columns exist
    expected_cols = [
        "total_transaction_amount",
        "average_transaction_amount",
        "transaction_count",
        "std_transaction_amount"
    ]
    for col in expected_cols:
        assert col in transformed.columns

    # Check that aggregation values are correct for CustomerId "C1"
    c1 = transformed[transformed["CustomerId"] == "C1"].iloc[0]
    assert c1["total_transaction_amount"] == 250
    assert np.isclose(c1["average_transaction_amount"], 125)
    assert c1["transaction_count"] == 2

# =====================================================
# 3️⃣ Test DateFeatureExtractor
# =====================================================
def test_date_feature_extractor(sample_data):
    date_extractor = DateFeatureExtractor()
    transformed = date_extractor.transform(sample_data)

    # Check that date columns exist
    for col in ["transaction_hour", "transaction_day", "transaction_month", "transaction_year"]:
        assert col in transformed.columns

    # Check that the values are correctly extracted
    first_row = transformed.iloc[0]
    assert first_row["transaction_hour"] == 10
    assert first_row["transaction_day"] == 1
    assert first_row["transaction_month"] == 1
    assert first_row["transaction_year"] == 2024

# =====================================================
# 4️⃣ Test ModelReadyDataProcessor returns all expected columns
# =====================================================
def test_model_ready_data_processor(sample_data):
    y = pd.Series([0, 1, 0, 1])  # Dummy target
    processor = ModelReadyDataProcessor()
    processor.fit(sample_data, y)
    transformed = processor.transform(sample_data)

    # Check that numeric + aggregated + date + OHE + WoE columns exist
    expected_columns = (
        ["Value"] +
        ["total_transaction_amount", "average_transaction_amount", "transaction_count", "std_transaction_amount"] +
        ["transaction_hour", "transaction_day", "transaction_month", "transaction_year"] +
        list(processor.ohe.get_feature_names_out(processor.ohe.feature_names_in_)) +
        [f"{c}_woe" for c in processor.woe_maps_.keys()]
    )
    assert set(expected_columns) == set(transformed.columns)
