from pydantic import BaseModel
from typing import List, Literal
from enum import Enum

# Enforce strict choices for the API
class RentalType(str, Enum):
    MONTHLY = "MONTHLY"
    DAILY = "DAILY"

# --- Heatmap Data Models ---
class HeatmapPoint(BaseModel):
    neighborhood: str
    latitude: float
    longitude: float
    current_avg_price: float
    trend_status: Literal["HOT", "COOL", "STABLE"]
    trend_description: str

class HeatmapResponse(BaseModel):
    rental_type: RentalType
    data: List[HeatmapPoint]

# --- Dashboard Chart Data Models ---
class DataPoint(BaseModel):
    date: str
    price: float
    type: Literal["historical", "forecast"]

class ForecastResponse(BaseModel):
    neighborhood: str
    rental_type: RentalType
    trend_status: str
    growth_rate_projection: float
    chart_data: List[DataPoint]