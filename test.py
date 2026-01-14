import cv2
import numpy as np
import joblib
import pickle

# Quick test function
def test_single_image(image_path):
    """Test one image quickly"""
    # Load model
    model = joblib.load('models/currency_model.joblib')
    with open('models/classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    
    # Load and process image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten() / 255.0
    
    # Predict
    pred_idx = model.predict([img])[0]
    prob = model.predict_proba([img])[0][pred_idx]
    
    print(f"\nImage: {image_path}")
    print(f"Predicted: {classes[pred_idx]}")
    print(f"Confidence: {prob:.1%}")
    
    return classes[pred_idx]

# Usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_single_image(sys.argv[1])
    else:
        print("Usage: python test.py <image_path>")

