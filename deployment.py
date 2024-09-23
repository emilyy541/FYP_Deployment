import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

# Load the saved Random Forest models for current prediction
with open('random_forest_model.pkl', 'rb') as file:
    rf_models = pickle.load(file)

# Load the LSTM model for time series prediction
lstm_model = tf.keras.models.load_model('lstm_model.h5')

# Streamlit app interface
st.title('Early Detection of Nutrient Pollution in Gulf of Alaska')

st.write("""
    ### Enter the values for the following variables:
""")

# Location selection for site
site = st.selectbox('Select Site Location', ['Homer', 'Seldovia'])

# Inputs for the feature variables (using number_input to ensure they are floats)
feature_5 = st.number_input('Temperature (°C)', value=0.0, step=0.1)         
feature_6 = st.number_input('Salinity (Sal)', value=0.0, step=0.1)           
feature_7 = st.number_input('Dissolved Oxygen (mg/L)', value=0.0, step=0.1) 
feature_8 = st.number_input('Depth (m)', value=0.0, step=0.1)                
feature_9 = st.number_input('pH', value=0.0, step=0.1)                       
feature_10 = st.number_input('Turbidity (NTU)', value=0.0, step=0.1)         
feature_11 = st.number_input('Chlorophyll Fluorescence', value=0.0, step=0.1) 

# Placeholder values for engineered features
feature_12 = 0.0  
feature_13 = 0.0  

# Convert location to numerical value if needed (encode location)
location_mapping = {'Homer': 1, 'Seldovia': 0}
location_feature = location_mapping[site]

# Combine all features into an array
input_features = np.array([[feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11,
                            feature_12, feature_13, location_feature]])

# Define the threshold values (as required)
thresholds = {
    'orthophosphate': 0.030,  # Adjusted based on 75th percentile
    'ammonium': 0.028,        # Adjusted based on 75th percentile
    'nitrite_nitrate': 0.158, # Adjusted based on 75th percentile
    'chlorophyll': 1.865      # Adjusted based on 75th percentile
}

# Pollution Classification Logic
def classify_variable_level(value, variable):
    """Classify each variable based on its level with dynamic adjustment."""
    if value <= thresholds[variable]:  # Light Pollution
        return "Light"
    elif value <= thresholds[variable] * 1.2: # Custom range for Moderate
        return "Moderate"
    else:  # Heavy Pollution
        return "Heavy"


def classify_overall_pollution(individual_status):
    """Classify overall nutrient pollution based on all variables."""
    if "Heavy" in individual_status.values():
        return "Heavy"
    elif list(individual_status.values()).count("Moderate") >= 2:
        return "Moderate"
    else:
        return "Light"

# Function to display alert notifications based on pollution classification
def display_alert_notification(overall_pollution):
    if overall_pollution == "Light":
        st.success("""Light Pollution: The pollution levels are low, but it is essential to maintain monitoring 
                      to protect marine ecosystems and ensure the sustainability of coastal environments. 
                      Preserving healthy water quality is crucial for supporting marine life and the well-being 
                      of coastal communities.""")
    elif overall_pollution == "Moderate":
        st.warning("""Moderate Pollution: Pollution levels are moderate, indicating a potential risk to marine biodiversity 
                      and coastal habitats. Implementing precautionary measures now can help prevent further degradation 
                      and support the resilience of marine ecosystems as well as the livelihoods that depend on them.""")
    elif overall_pollution == "Heavy":
        st.error("""Heavy Pollution: Warning! Pollution levels are high! Immediate action is required to mitigate environmental 
                    risks and prevent severe impacts on marine life and coastal resources. Addressing this issue is crucial 
                    for preserving the health of our oceans and the communities that rely on them for sustenance 
                    and economic activities.""")
        
# Store input history and display history table
if 'history' not in st.session_state:
    st.session_state.history = []

# Button to make current pollution prediction
if st.button('Predict Current Levels'):
    st.subheader(f'Predicted Nutrient Pollution Levels:')
    
    # Predict current nutrient pollution levels using Random Forest models
    predictions = {}
    individual_status = {}
    
    # Debugging: Print the input features
    print("Input Features: ", input_features)
    
    for target in ['orthophosphate', 'ammonium', 'nitrite_nitrate', 'chlorophyll']:
        # Make predictions for each target variable
        predictions[target] = rf_models[target].predict(input_features)[0]
        
        # Debugging: Print predictions
        print(f"Prediction for {target}: {predictions[target]}")
        
        # Classify the prediction based on thresholds
        individual_status[target] = classify_variable_level(predictions[target], target)
    
    # Display individual predictions and levels
    for target, level in individual_status.items():
        st.write(f"**{target.capitalize()} (mg/L):** {predictions[target]:.2f} - {level}")
    
    # Debugging: Check the individual classification status
    print("Individual Status: ", individual_status)
    
    # Classify overall pollution based on individual status
    overall_pollution = classify_overall_pollution(individual_status)
    
    # Debugging: Print the overall pollution level
    print("Overall Pollution: ", overall_pollution)
    
    # Store prediction in history
    current_prediction = {
        'Site': site,
        'Temperature (°C)': feature_5,
        'Salinity (Sal)': feature_6,
        'Dissolved Oxygen (mg/L)': feature_7,
        'Depth (m)': feature_8,
        'pH': feature_9,
        'Turbidity (NTU)': feature_10,
        'Chlorophyll Fluorescence': feature_11,
        'Orthophosphate': predictions['orthophosphate'],
        'Ammonium': predictions['ammonium'],
        'Nitrite/Nitrate': predictions['nitrite_nitrate'],
        'Chlorophyll': predictions['chlorophyll'],
        'Overall Pollution': overall_pollution
    }
    st.session_state.history.append(current_prediction)

    # Show input history as a table
    st.write("User Input History:")
    st.write(pd.DataFrame(st.session_state.history))

    # Display alert notifications
    display_alert_notification(overall_pollution)

    # Graphical representation
    fig, ax = plt.subplots()
    nutrients = ['Orthophosphate', 'Ammonium', 'Nitrite/Nitrate', 'Chlorophyll']
    values = [predictions['orthophosphate'], predictions['ammonium'], predictions['nitrite_nitrate'], predictions['chlorophyll']]
    thresholds_list = [thresholds['orthophosphate'], thresholds['ammonium'], thresholds['nitrite_nitrate'], thresholds['chlorophyll']]

    ax.bar(nutrients, values, color='blue', label='Predicted Values')
    for nutrient, threshold in zip(nutrients, thresholds_list):
        ax.axhline(y=threshold, linestyle='--', label=f'{nutrient} Threshold')

    ax.set_ylabel('Concentration (mg/L)')
    ax.set_title(f'Nutrient Pollution Levels vs Thresholds')
    ax.legend()
    st.pyplot(fig)

# Time Series Prediction using LSTM with adjustable years
num_years = st.slider('Select Number of Years for Prediction', min_value=1, max_value=4, value=4)

if st.button(f'Prediction of Nutrient Pollution Levels in Next {num_years} Years'):
    st.subheader(f'Time Series Predictions for Nutrient Pollution for Next {num_years} Years')

    # Prepare the input for LSTM (reshape as required by LSTM input)
    lstm_input = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))

    # Predict the next 'num_years' using LSTM
    lstm_predictions = lstm_model.predict(lstm_input)
    
    # Generate years for x-axis based on the number of years selected
    years = np.arange(2022, 2022 + num_years)  # Adjust years based on 'num_years'

    # Plot the time series predictions
    fig, ax = plt.subplots()
    ax.plot(years, lstm_predictions.flatten()[:num_years], marker='o', label='Predicted Pollution Level')
    ax.set_xlabel('Year')
    ax.set_xticks(years)  # Set x-axis ticks to display whole years only
    ax.set_ylabel('Nutrient Pollution Level (mg/L)')
    ax.set_title(f'Predicted Pollution Levels Over the Next {num_years} Years')
    st.pyplot(fig)
