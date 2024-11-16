import streamlit as st
import joblib
import numpy as np

# Load pre-trained model and scaler
model = joblib.load('knn_model.joblib')
scaler = joblib.load('Models/scaler.joblib')

# Function for preprocessing input features
def preprocessing(age, appearance, goals, minutes_played, highest_valuated_price_euro, price_category):
    # Create a feature dictionary based on the input data
    dict_f = {
        'age': age,
        'appearance': appearance,
        'goals': goals,
        'minutes_played': minutes_played,
        'Highest_valuated_price_euro': highest_valuated_price_euro,
        'price_category_Premium': price_category == 'Premium',
        'price_category_Mid': price_category == 'Mid',
        'price_category_Budget': price_category == 'Budget'
    }

    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    return features_list

# Streamlit UI
st.title("Football Player Prediction Model")
st.write("Welcome to the prediction app! Enter the details below to make a prediction.")

# Input fields for the user
age = st.number_input("Age", min_value=18, max_value=40, step=1)
appearance = st.number_input("Appearance", min_value=0, max_value=500, step=1)
goals = st.number_input("Goals", min_value=0, max_value=300, step=1)
minutes_played = st.number_input("Minutes Played", min_value=0, max_value=50000, step=1)
highest_valuated_price_euro = st.number_input("Highest Valuated Price (Euro)", min_value=0.0, step=0.1)
price_category = st.selectbox("Price Category", options=["Premium", "Mid", "Budget"])

# When the user clicks on the "Predict" button
if st.button("Predict"):
    # Preprocess the input data
    data = preprocessing(age, appearance, goals, minutes_played, highest_valuated_price_euro, price_category)

    # Scale the input data
    scaled_data = scaler.transform([data])

    # Make the prediction using the pre-trained model
    y_pred = model.predict(scaled_data)

    # Display the prediction result
    st.write(f"Prediction: {y_pred[0]}")
