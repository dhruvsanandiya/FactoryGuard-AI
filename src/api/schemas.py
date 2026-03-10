"""Pydantic schemas for API request/response validation"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint"""
    machine_id: str = Field(..., description="Machine identifier")
    temperature: float = Field(..., description="Temperature sensor reading")
    pressure: float = Field(..., description="Pressure sensor reading")
    vibration: float = Field(..., description="Vibration sensor reading")
    
    @validator('temperature', 'pressure', 'vibration')
    def validate_sensor_values(cls, v):
        """Validate sensor values are within reasonable ranges"""
        if not isinstance(v, (int, float)):
            raise ValueError("Sensor values must be numeric")
        if abs(v) > 100000:  # Very permissive range
            raise ValueError("Sensor value out of acceptable range")
        return float(v)


class RiskFactor(BaseModel):
    """Individual risk factor contribution"""
    feature: str
    value: float
    shap_value: float
    explanation: str


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint"""
    failure_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted failure probability")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, or HIGH")
    top_risk_factors: List[str] = Field(..., description="Top contributing risk factors")
    shap_explanations: Optional[List[RiskFactor]] = Field(None, description="Detailed SHAP explanations")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None

