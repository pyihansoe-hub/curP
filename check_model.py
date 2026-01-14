import joblib
import pickle
import numpy as np

model = joblib.load('models/currency_model.joblib')
with open('models/classes.pkl', 'rb') as f:
    classes = pickle.load(f)

print("Model classes:", classes)
print("Number of classes:", len(classes))

# Test with dummy data
test_data = np.random.rand(10000)  # 100x100 flattened
test_data = test_data.reshape(1, -1)

try:
    prediction = model.predict(test_data)[0]
    confidence = model.predict_proba(test_data)[0][prediction]
    print(f"Test prediction: {classes[prediction]} ({confidence:.2%})")
except Exception as e:
    print("Error:", e)