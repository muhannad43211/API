from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for specific origins (you can adjust these as needed)
origins = [
    "http://localhost:3000",  # Streamlit (or any frontend app)
    "https://your-frontend-app.com",
]

# Add the CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows the specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load model and scaler (adjust paths if necessary)
model = joblib.load('knn_model.joblib')
scaler = joblib.load('Models/scaler.joblib')

# Define Pydantic model for input data
class InputFeatures(BaseModel):
    age: int
    appearance: int
    goals: int
    minutes_played: int
    Highest_valuated_price_euro: float
    price_category: str

# Prediction endpoint
@app.post("/predict")
async def get_prediction(input_features: InputFeatures):
    dict_f = {
        'age': input_features.age,
        'appearance': input_features.appearance,
        'goals': input_features.goals,
        'minutes_played': input_features.minutes_played,
        'Highest_valuated_price_euro': input_features.Highest_valuated_price_euro,
        'price_category_Premium': input_features.price_category == 'Premium',
        'price_category_Mid': input_features.price_category == 'Mid',
        'price_category_Budget': input_features.price_category == 'Budget'
    }

    features_list = [dict_f[key] for key in sorted(dict_f)]

    # Scale features and make prediction
    scaled_data = scaler.transform([features_list])
    y_pred = model.predict(scaled_data)

    return {"pred": y_pred.tolist()[0]}
