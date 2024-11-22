import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read the dataset
data = pd.read_csv("user_behavior_dataset.csv")

# Preprocessing Using label Encoder
le_device = LabelEncoder()
le_os = LabelEncoder()
le_gender = LabelEncoder()

data['Device Model'] = le_device.fit_transform(data['Device Model'])
data['Operating System'] = le_os.fit_transform(data['Operating System'])
data['Gender'] = le_gender.fit_transform(data['Gender'])

# Drop 'User ID' column
data = data.drop('User ID', axis=1)

# Define features (X) and target (y)
X = data.drop(columns='User Behavior Class', axis=1)
y = data['User Behavior Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy Calculation
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# please Given Prediction based on New data
# Prediction based on new data
new_data = [[4,1,187,4.3,1367,58,988,31,0]]  # Example of new data (replace with your data)
prediction = model.predict(new_data)
print(f"Prediction: {prediction}")