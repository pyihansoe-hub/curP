import cv2
import numpy as np
import joblib
import pickle
import os

model = joblib.load('models/currency_model.joblib')
with open('models/classes.pkl', 'rb') as f:
    classes = pickle.load(f)

print("DEBUG MODE")
print("Classes:", classes)

def predict_debug(img_path):
    print(f"\nAnalyzing: {img_path}")
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print("ERROR: Cannot read image")
        return
    
    print(f"Original size: {img.shape}")
    
    # Step by step preprocessing
    resized = cv2.resize(img, (100, 100))
    print(f"Resized: {resized.shape}")
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale: {gray.shape}")
    
    flattened = gray.flatten()
    print(f"Flattened: {flattened.shape}, min={flattened.min()}, max={flattened.max()}")
    
    normalized = flattened / 255.0
    print(f"Normalized: min={normalized.min():.3f}, max={normalized.max():.3f}")
    
    # Predict
    pred_idx = model.predict([normalized])[0]
    prob = model.predict_proba([normalized])[0]
    
    print(f"\nAll probabilities:")
    for i, p in enumerate(prob):
        print(f"  {classes[i]}: {p:.4f}")
    
    print(f"\nPredicted: {classes[pred_idx]} ({prob[pred_idx]:.2%})")
    
    # Show image
    display = cv2.resize(img, (400, 300))
    cv2.putText(display, f"True: ???", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display, f"Pred: {classes[pred_idx]}", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Conf: {prob[pred_idx]:.1%}", (20, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Debug', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test
while True:
    path = input("\nImage path (or 'quit'): ")
    if path.lower() == 'quit':
        break
    if os.path.exists(path):
        predict_debug(path)
    else:
        print("File not found")