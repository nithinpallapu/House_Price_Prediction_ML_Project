import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
with open('house_price_model (1).pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 India House Price Prediction")
st.write("Enter property details to get an estimated house price.")

st.divider()

# --------------------------------------------------
# State → City Mapping (VALID & REALISTIC)
# --------------------------------------------------
state_city_map = {
    'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Salem'],
    'Delhi': ['New Delhi', 'Dwarka', 'Rohini', 'Saket'],
    'Telangana': ['Hyderabad', 'Warangal', 'Nizamabad', 'Karimnagar'],
    'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot'],
    'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Nashik'],
    'Karnataka': ['Bangalore', 'Mysore', 'Mangalore', 'Hubli']
}

# --------------------------------------------------
# Dropdown options
# --------------------------------------------------
property_type_options = ['Apartment', 'Independent House', 'Villa']
furnished_status_options = ['Unfurnished', 'Semi-Furnished', 'Furnished']
public_transport_options = ['Low', 'Medium', 'High']
amenities_options = ['Garden', 'Clubhouse', 'Pool', 'Gym']
facing_options = ['North', 'East', 'South', 'West']

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
user_input = {}

# State & City (Dependent)
user_input['State'] = st.selectbox(
    'State',
    list(state_city_map.keys())
)

user_input['City'] = st.selectbox(
    'City',
    state_city_map[user_input['State']]
)

user_input['Property_Type'] = st.selectbox(
    'Property Type',
    property_type_options
)

user_input['BHK'] = st.slider(
    'BHK (Bedrooms)',
    min_value=1,
    max_value=5,
    value=2
)

user_input['Size_in_SqFt'] = st.number_input(
    'Size in SqFt',
    min_value=100,
    max_value=5000,
    value=1200
)

user_input['Furnished_Status'] = st.selectbox(
    'Furnished Status',
    furnished_status_options
)

user_input['Total_Floors'] = st.slider(
    'Total Floors in Building',
    min_value=1,
    max_value=30,
    value=15
)

user_input['Age_of_Property'] = st.slider(
    'Age of Property (Years)',
    min_value=0,
    max_value=35,
    value=10
)

user_input['Nearby_Schools'] = st.slider(
    'Nearby Schools',
    min_value=1,
    max_value=10,
    value=5
)

user_input['Nearby_Hospitals'] = st.slider(
    'Nearby Hospitals',
    min_value=1,
    max_value=10,
    value=5
)

user_input['Public_Transport_Accessibility'] = st.selectbox(
    'Public Transport Accessibility',
    public_transport_options
)

user_input['Parking_Space'] = st.checkbox(
    'Parking Space Available'
)

user_input['Security'] = st.checkbox(
    'Security Available'
)

user_input['Amenities'] = st.selectbox(
    'Amenities',
    amenities_options
)

user_input['Facing'] = st.selectbox(
    'Facing',
    facing_options
)

# --------------------------------------------------
# Convert Input to DataFrame
# --------------------------------------------------
input_df = pd.DataFrame([user_input])

# Convert boolean fields to categorical values
input_df['Parking_Space'] = input_df['Parking_Space'].map({True: 'true', False: 'false'})
input_df['Security'] = input_df['Security'].map({True: 'true', False: 'false'})

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button('Predict Price'):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted House Price: ₹ {prediction:,.2f} Lakhs")
    except Exception as e:
        st.error("Prediction failed. Please check input values.")
        st.write(e)

st.markdown("---")
st.info(
    "Disclaimer: This is a model-based estimate using a real-world–inspired dataset. "
    "Actual market prices may vary."
)
