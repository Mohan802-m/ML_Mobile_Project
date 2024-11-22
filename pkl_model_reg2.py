
import pickle 
 # Load the model from the file
with open('user_behavior_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

new_data = [[4,1,187,4.3,1367,58,988,31,0]]  # Example of new data (replace with your data)
prediction = loaded_model.predict(new_data)
print(f"Prediction: {prediction}")
