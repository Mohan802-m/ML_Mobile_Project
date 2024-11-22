import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Title of the Streamlit app
st.title('User Behavior Classification Model')

# File uploader widget to upload a dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If a file is uploaded, proceed with the processing
if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", data.head())  # Display the first few rows of the dataset

    # Preprocessing Using Label Encoder
    le = LabelEncoder()

    if 'Device Model' in data.columns:
        data['Device Model'] = le.fit_transform(data['Device Model'])

    if 'Operating System' in data.columns:
        data['Operating System'] = le.fit_transform(data['Operating System'])

    if 'Gender' in data.columns:
        data['Gender'] = le.fit_transform(data['Gender'])

    # Drop 'User ID' if it exists
    if 'User ID' in data.columns:
        data = data.drop('User ID', axis=1)

    # Define features (X) and target (y)
    if 'User Behavior Class' in data.columns:
        X = data.drop(columns='User Behavior Class', axis=1)
        y = data['User Behavior Class']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy (R2 score)
        accuracy = r2_score(y_test, y_pred)

        # Display results
        st.write(f"Accuracy (R2 Score): {accuracy * 100:.2f}%")

        # Display a button to show detailed results
        if st.button('Show Predictions'):
            predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            st.write(predictions)

    else:
        st.write("The dataset does not contain a 'User Behavior Class' column.")
