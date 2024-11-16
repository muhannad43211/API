from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# Welcome message at the root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to Tuwaiq Academy"}

# Load the pre-trained model and scaler
model = joblib.load('knn_model.joblib')
scaler = joblib.load('Models/scaler.joblib')

# Define the Pydantic model for input data validation
class InputFeatures(BaseModel):
    age: int
    appearance: int
    goals: int
    minutes_played: int
    Highest_valuated_price_euro: float
    price_category: str

# Preprocessing function to transform input features
def preprocessing(input_features: InputFeatures):
    dict_f = {
        'age': input_features.age,
        'appearance': input_features.appearance,
        'goals': input_features.goals,
        'minutes_played': input_features.minutes_played,
        'price_category_Premium': input_features.price_category == 'Premium',
        'price_category_Mid': input_features.price_category == 'Mid',
        'price_category_Budget': input_features.price_category == 'Budget'
    }

    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    # Return the features list for scaling and prediction
    return features_list

@app.get("/predict")
def get_prediction(input_features: InputFeatures):
    # Preprocess the input features
    features = preprocessing(input_features)
    
    # Scale the input features using the pre-loaded scaler
    scaled_features = scaler.transform([features])

    # Make a prediction using the pre-trained model
    y_pred = model.predict(scaled_features)
    
    # Return the prediction result
    return {"pred": y_pred.tolist()[0]}

@app.post("/predict")
async def post_prediction(input_features: InputFeatures):
    # Preprocess the input features
    features = preprocessing(input_features)
    
    # Scale the input features using the pre-loaded scaler
    scaled_features = scaler.transform([features])

    # Make a prediction using the pre-trained model
    y_pred = model.predict(scaled_features)
    
    # Return the prediction result
    return {"pred": y_pred.tolist()[0]}
