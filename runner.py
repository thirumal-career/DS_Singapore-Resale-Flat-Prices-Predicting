import streamlit as st
import pickle
import numpy as np


with open("house_price_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Feature names
features = ['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model',
             'year', 'month_of_year', 'lease_commence_year',
            'remaining_lease_years', 'remaining_lease_months']

# Categorical variable mappings
categorical_mappings = {
    'town': {'SENGKANG': 20, 'PUNGGOL': 17, 'WOODLANDS': 24, 'YISHUN': 25,
             'TAMPINES': 22, 'JURONG WEST': 13, 'BEDOK': 1, 'HOUGANG': 11,
             'CHOA CHU KANG': 8, 'ANG MO KIO': 0, 'BUKIT MERAH': 4, 'BUKIT PANJANG': 5,
             'BUKIT BATOK': 3, 'TOA PAYOH': 23, 'PASIR RIS': 16, 'KALLANG/WHAMPOA': 14,
             'QUEENSTOWN': 18, 'SEMBAWANG': 19, 'GEYLANG': 10, 'CLEMENTI': 9,
             'JURONG EAST': 12, 'BISHAN': 2, 'SERANGOON': 21, 'CENTRAL AREA': 7,
             'MARINE PARADE': 15, 'BUKIT TIMAH': 6},
    
    'flat_type': {'4 ROOM': 3, '5 ROOM': 4, '3 ROOM': 2,
                  'EXECUTIVE': 5, '2 ROOM': 1, 'MULTI-GENERATION': 6,
                  '1 ROOM': 0},
    
    'storey_range': {'04 TO 06': 1, '07 TO 09': 2, '10 TO 12': 3, '01 TO 03': 0,
                     '13 TO 15': 4, '16 TO 18': 5, '19 TO 21': 6, '22 TO 24': 7,
                     '25 TO 27': 8, '28 TO 30': 9, '31 TO 33': 10, '34 TO 36': 11,
                     '37 TO 39': 12, '40 TO 42': 13, '43 TO 45': 14, '46 TO 48': 15,
                     '49 TO 51': 16},
    
    'flat_model': {'Model A': 8, 'Improved': 5, 'New Generation': 12, 'Premium Apartment': 13,
                   'Simplified': 16, 'Apartment': 3, 'Maisonette': 7, 'Standard': 17,
                   'DBSS': 4, 'Model A2': 10, 'Model A-Maisonette': 9, 'Adjoined flat': 2,
                   'Type S1': 19, 'Type S2': 20, 'Premium Apartment Loft': 14, 'Terrace': 18,
                   'Multi Generation': 11, '2-room': 0, 'Improved-Maisonette': 6, '3Gen': 1,
                   'Premium Maisonette': 15},
}


st.title("House Price Prediction App")

input_data = {}
for feature in features:
    if feature in categorical_mappings:
        selected_option = st.sidebar.selectbox(f"Select {feature.capitalize()}:", options=list(categorical_mappings[feature].keys()))
        input_data[feature] = categorical_mappings[feature][selected_option]

        input_data[feature] = st.sidebar.number_input(f"{feature.capitalize()}:")
    else:
        input_data[feature] = st.sidebar.number_input(f"{feature.capitalize()}:")
    
# Make predictions using the loaded model
if st.sidebar.button("Predict"):
    input_array = np.array([input_data[feature] for feature in features]).reshape(1, -1)
    prediction = model.predict(input_array)

    # Display the prediction result
    prediction_scale = np.exp(prediction[0])
    st.subheader("Prediction Result:")
    st.write(f"The predicted house price is: {prediction_scale:,.2f} INR")
