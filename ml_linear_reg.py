# Read data
import pandas as pd
data = pd.read_csv("user_behavior_dataset.csv")
data
#Preprocessing Using label Encoder for model col ------------ 1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Device Model']=le.fit_transform(data['Device Model'])
#label Encoder for OS --2
data['Operating System']=le.fit_transform(data['Operating System'])
#label Encoder for gender --3
data['Gender']=le.fit_transform(data['Gender'])
data=data.drop('User ID',axis=1)
# Define features (X) and target (y)
X=data.drop(columns='User Behavior Class',axis=1)
y=data['User Behavior Class']
# Spplit and Train the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred
# Accurecy Calculation
from sklearn.metrics import r2_score
# Accuracy score
accuracy = r2_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Prediction based on new data
new_data = [[4,1,187,4.3,1367,58,988,31,0]]  # Example of new data (replace with your data)
prediction = model.predict(new_data)
print(f"Prediction: {prediction}")
