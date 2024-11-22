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

device_model = st.number_input("Device Model (Encoded)", min_value=0, max_value=4,step=1)
operating_system = st.number_input("Operating System (Encoded)", min_value=0, max_value=1,step=1)
page_views = st.number_input("Page Views", min_value=0.0, step=1.0)
session_duration = st.number_input("Session Duration (mins)", min_value=0.0, step=0.1)
actions_per_session = st.number_input("Actions Per Session", min_value=0.0, step=0.1)
click_through_rate = st.number_input("Click Through Rate (%)", min_value=0.0, step=0.1)
avg_scroll_depth = st.number_input("Average Scroll Depth (pixels)", min_value=0.0, step=1.0)
days_since_last_visit = st.number_input("Days Since Last Visit", min_value=0, step=1)
gender = st.number_input("Gender (Encoded)", min_value=0, max_value=1, step=1)

# Collect new data for prediction
new_data = [[
    device_model, operating_system, page_views, session_duration,
    actions_per_session, click_through_rate, avg_scroll_depth,
    days_since_last_visit, gender
]]

# Make predictions
if st.button("Predict"):
    prediction = model.predict(new_data)
    st.success(f"Predicted User Behavior Class: {prediction[0]:.2f}")
