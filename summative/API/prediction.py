from fastapi import FastAPI, HTTPException  # type: ignore
from pydantic import BaseModel, Field, validator  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
import joblib  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from enum import Enum

# Load the trained model and reference feature names
model = joblib.load('./model/best_model.pkl')
X_encoded_columns = model.feature_names_in_

app = FastAPI(title="Crop Yield Prediction API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Valid crop types
class CropType(str, Enum):
    maize = "Maize"
    potatoes = "Potatoes"
    rice = "Rice, paddy"
    sorghum = "Sorghum"
    soybeans = "Soybeans"
    wheat = "Wheat"
    cassava = "Cassava"
    sweet_potatoes = "Sweet potatoes"
    plantains = "Plantains"
    yams = "Yams"

# Pydantic model for request validation with constraints in Swagger
class PredictionRequestRanges(BaseModel):
    crop_type: CropType = Field(
        ..., 
        description="Type of crop. Choose from: 'Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', etc."
    )
    year: int = Field(
        ..., 
        ge=1990, le=2050, 
        description="Year of prediction. Must be between 1990 and 2050."
    )
    avg_rainfall: float = Field(
        ..., 
        ge=51, le=3240, 
        description="Average annual rainfall in mm. Must be between 51 and 3240."
    )
    pesticides: float = Field(
        ..., 
        ge=0.04, le=36778, 
        description="Pesticides used in tonnes. Must be between 0.04 and 36778."
    )
    avg_temp: float = Field(
        ..., 
        ge=1.3, le=30.65, 
        description="Average temperature in Celsius. Must be between 1.3 and 30.65."
    )

    # Additional validation logic (optional if ranges are already enforced by Field constraints)
    @validator("year")
    def validate_year(cls, value):
        if not (1990 <= value <= 2050):
            raise ValueError("Year must be between 1990 and 2050.")
        return value

    @validator("avg_rainfall")
    def validate_avg_rainfall(cls, value):
        if not (51 <= value <= 3240):
            raise ValueError("Average rainfall must be between 51 and 3240 mm.")
        return value

    @validator("pesticides")
    def validate_pesticides(cls, value):
        if not (0.04 <= value <= 36778):
            raise ValueError("Pesticides must be between 0.04 and 36778 tonnes.")
        return value

    @validator("avg_temp")
    def validate_avg_temp(cls, value):
        if not (1.3 <= value <= 30.65):
            raise ValueError("Average temperature must be between 1.3 and 30.65 °C.")
        return value


@app.post('/predict')
async def predict(data: PredictionRequestRanges):
    try:
        # Prepare input DataFrame
        input_data = pd.DataFrame({
            'Year': [data.year],
            'average_rain_fall_mm_per_year': [data.avg_rainfall],
            'pesticides_tonnes': [data.pesticides],
            'avg_temp': [data.avg_temp]
        })

        # One-hot encode the crop type
        crop_dummies = pd.get_dummies(pd.Series([data.crop_type]), prefix='Item')
        input_data = pd.concat([input_data, crop_dummies], axis=1)

        # Ensure all required columns exist
        for col in X_encoded_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[X_encoded_columns]

        # Make a prediction
        yield_prediction = model.predict(input_data)[0]
        return {"predicted_yield": str(yield_prediction) + " hg/ha"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000)
