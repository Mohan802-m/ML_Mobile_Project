import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Title and Description
st.title("User Behavior Prediction App")
st.markdown("""
This app predicts the **User Behavior Class** based on features like Device Model, Operating System, and more.
Fill in the feature values below and click the **Predict** button.
""")

# Load and preprocess dataset
@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv("user_behavior_dataset.csv")
    
    # Preprocess using LabelEncoder
    le_device = LabelEncoder()
    le_os = LabelEncoder()
    le_gender = LabelEncoder()

    data['Device Model'] = le_device.fit_transform(data['Device Model'])
    data['Operating System'] = le_os.fit_transform(data['Operating System'])
    data['Gender'] = le_gender.fit_transform(data['Gender'])
    
    # Drop 'User ID' column
    data = data.drop('User ID', axis=1)
    
    # Split data
    X = data.drop(columns='User Behavior Class', axis=1)
    y = data['User Behavior Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Train model
@st.cache_resource
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Load data and train model
X_train, X_test, y_train, y_test = load_and_preprocess_data()
model = train_model(X_train, y_train)

# Calculate and display model accuracy
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
st.sidebar.markdown(f"### Model Accuracy: {accuracy * 100:.2f}%")

# Input features for prediction
st.header("Enter Feature Values for Prediction")

# STEP 1 : Select Device Name & Its Automatically Encoading....
#############################################################################################################
# Define the mapping of numeric encodings to labels
device_mapping = {
    0: "Xiaomi Mi 11 ",
    1: "iPhone 12",
    2: "Google Pixel 5",
    3: "OnePlus 9",
    4: "Samsung Galaxy S21",
}
# Reverse the mapping for encoding
reverse_device_mapping = {v: k for k, v in device_mapping.items()}
# Dropdown for selecting the device model
selected_label = st.selectbox("Select Device Model:", list(device_mapping.values()))
# Get the numeric encoding for the selected label
encoded_device_model = reverse_device_mapping[selected_label]
# Display the selected value and its encoding
st.write(f"Selected Device Model: {selected_label} (Encoded as: {encoded_device_model})")
##################################################################################################################

# STEP 2 : OS Selection...
os_mapping = {
    0: "Android",
    1: "iOS"
}
# Create a dropdown for Operating System
selected_os_label = st.selectbox("Select Operating System:", options=list(os_mapping.values()))
# Get the numeric encoding for the selected label
encoded_os = list(os_mapping.keys())[list(os_mapping.values()).index(selected_os_label)]
# Display the result
st.write(f"Selected Operating System: {selected_os_label} (Encoded as: {encoded_os})")

###########################################################################################################
page_views = st.number_input("Page Views", min_value=0.0, step=1.0)

session_duration = st.number_input("Session Duration (mins)", min_value=0.0, step=0.01)

actions_per_session = st.number_input("Actions Per Session", min_value=0.0, step=0.1)

click_through_rate = st.number_input("Click Through Rate (%)", min_value=0.0, step=0.1)

avg_scroll_depth = st.number_input("Average Scroll Depth (pixels)", min_value=0.0, step=1.0)

days_since_last_visit = st.number_input("Days Since Last Visit", min_value=0, step=1)
####################################################################################################################
gender_mapping = {
    1: "Male",
    0: "Female"
}
# Create radio buttons for gender selection
selected_gender_label = st.radio("Select Gender:", options=list(gender_mapping.values()))
# Get the numeric encoding for the selected label
encoded_gender = list(gender_mapping.keys())[list(gender_mapping.values()).index(selected_gender_label)]
# Display the result
st.write(f"Selected Gender: {selected_gender_label} (Encoded as: {encoded_gender})")

# Collect new data for prediction
new_data = [[
    encoded_device_model, encoded_os, page_views, session_duration,
    actions_per_session, click_through_rate, avg_scroll_depth,
    days_since_last_visit, encoded_gender
]]

# Make predictions
if st.button("Predict"):
    prediction = model.predict(new_data)
    st.success(f"Predicted User Behavior Class: {prediction[0]:.2f}")

########################################################################################

# This is finall and accurate model to ready to deploy....... OK With Final Verification..

# Store the Model........

import pickle
# Save the trained model to a file
with open('user_mobile_behavior_model.pkl', 'wb') as f:
    pickle.dump(model, f)

