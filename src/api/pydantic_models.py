from pydantic import BaseModel
from typing import Dict

class PredictRequest(BaseModel):
    # All features required by your model
    Value: float
    total_transaction_amount: float
    average_transaction_amount: float
    transaction_count: float
    std_transaction_amount: float
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
    CurrencyCode_UGX: float
    # ... Add other one-hot and WoE features used by the model

class PredictResponse(BaseModel):
    risk_probability: float
