import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model
model = pickle.load(open('Finalizedk.pkl', 'rb'))

# Initialize StandardScaler
scaler = StandardScaler()

# Define Streamlit widgets for user input
st.title('K-Nearest Neighbors Model Prediction')

# Input fields with range constraints
inter_canine_distance_intraoral = st.number_input(
    'Inter Canine Distance Intraoral (mm)', 
    min_value=20.00, 
    max_value=40.00, 
    value=20.00
)
intercanine_distance_casts = st.number_input(
    'Intercanine Distance Casts (mm)', 
    min_value=6.00, 
    max_value=30.31, 
    value=6.00
)
right_canine_width_casts = st.number_input(
    'Right Canine Width Casts (mm)', 
    min_value=5.00, 
    max_value=11.00, 
    value=5.00
)
left_canine_width_casts = st.number_input(
    'Left Canine Width Casts (mm)', 
    min_value=5.00, 
    max_value=11.00, 
    value=5.00
)
right_canine_index_intra_oral = st.number_input(
    'Right Canine Index Intra Oral', 
    min_value=0.2, 
    max_value=0.4, 
    value=0.2
)
right_canine_index_casts = st.number_input(
    'Right Canine Index Casts', 
    min_value=0.2, 
    max_value=0.4, 
    value=0.2
)
left_canine_index_casts = st.number_input(
    'Left Canine Index Casts', 
    min_value=0.2, 
    max_value=0.4, 
    value=0.2
)

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'inter canine distance intraoral': [inter_canine_distance_intraoral],
    'intercanine distance casts': [intercanine_distance_casts],
    'right canine width casts': [right_canine_width_casts],
    'left canine width casts': [left_canine_width_casts],
    'right canine index intra oral': [right_canine_index_intra_oral],
    'right canine index casts': [right_canine_index_casts],
    'left canine index casts': [left_canine_index_casts]
})

# Display the entered values and their ranges
st.write('**Entered Values:**')
st.write(input_data)

# Display expected ranges
st.write('**Expected Ranges:**')
st.write('Inter Canine Distance Intraoral: 20 mm to 40 mm')
st.write('Intercanine Distance Casts: 6.00 mm to 30.31 mm')
st.write('Right Canine Width Casts: 5.00 mm to 11.00 mm')
st.write('Left Canine Width Casts: 5.00 mm to 11.00 mm')
st.write('Right Canine Index Intra Oral: 0.2 to 0.4')
st.write('Right Canine Index Casts: 0.2 to 0.4')
st.write('Left Canine Index Casts: 0.2 to 0.4')

# Predict button
if st.button('Predict'):
    # Scale the input data
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Predict using the loaded model
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)
    
    # Show prediction results
    if prediction[0] == 0:
        st.write('**Prediction:** Female')
    else:
        st.write('**Prediction:** Male')
    st.write('**Prediction Probability:**', prediction_proba[0])




