import joblib
import numpy as np

# Load saved files
model = joblib.load("../saved_models/best_model.pkl")
scaler = joblib.load("../saved_models/scaler.pkl")
label_encoder = joblib.load("../saved_models/label_encoder.pkl")

# Example test input
# Order must match dataset columns:
# Python, Java, JavaScript, SQL, DSA, Cloud, DevOps,
# Networking, DataAnalysis, Communication,
# CGPA, Projects, EasySolved, MediumSolved, HardSolved

sample_input = np.array([[3, 2, 2, 2, 3, 2, 2, 5, 2, 3, 7.8, 3, 40, 30, 15]])

# Scale input
scaled_input = scaler.transform(sample_input)

# Predict
prediction = model.predict(scaled_input)

# Convert back to role name
predicted_role = label_encoder.inverse_transform(prediction)

print("Predicted Role:", predicted_role[0])