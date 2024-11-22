import pickle
# Load the model from the pickle file
with open('user_mobile_behavior_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
# Now you can use the loaded_model as needed
print(type(loaded_model))  # Check the type of the loaded model

import os

file_path = 'user_mobile_behavior_model.pkl'
if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        loaded_model = pickle.load(f)
    print("Model loaded successfully.")
else:
    print(f"The file {file_path} does not exist.")

