import streamlit as st
import numpy as np
import pickle

# Load the saved linear regression model
model_file = 'WC_LR_Model.pkl'
with open(model_file, 'rb') as file:
    model_lr = pickle.load(file)

# Streamlit App Title and Description
st.set_page_config(page_title="Water Consumption Prediction", page_icon="💧", layout="centered")

st.title("💧 Water Consumption Prediction")
st.write("Enter the population and water production values to predict water consumption.")

# Input Fields
population = st.number_input(
    "Population (X1):",
    min_value=0.0, step=1.0, format="%.2f",
    help="Enter the population value, e.g., 40000."
)
water_production = st.number_input(
    "Water Production (X2):",
    min_value=0.0, step=1.0, format="%.2f",
    help="Enter the water production value, e.g., 18000."
)

# Prediction Button
if st.button("Predict"):
    try:
        # Create the feature array without scaling
        input_features = np.array([[population, water_production]])
        prediction = model_lr.predict(input_features)

        # Display the result
        st.success(f"🌊 Predicted Water Consumption: {prediction[0]:.2f} thousand litres")
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<footer style='text-align: center; color: gray;'>"
    "&copy; 2025 Water Consumption Predictor | Designed with Streamlit"
    "</footer>",
    unsafe_allow_html=True,
)
