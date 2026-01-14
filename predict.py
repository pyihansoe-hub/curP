import cv2
import numpy as np
import joblib
import pickle
import os

model = joblib.load('models/currency_model.joblib')
with open('models/classes.pkl', 'rb') as f:
    classes = pickle.load(f)

def predict_image(img):
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten() / 255.0
    pred_idx = model.predict([img])[0]
    prob = model.predict_proba([img])[0][pred_idx]
    return classes[pred_idx], prob

print("="*50)
print("CURRENCY RECOGNITION")
print("="*50)
print(f"Loaded {len(classes)} currencies: {classes}")

while True:
    print("\n1. Enter image path")
    print("2. Webcam")
    print("3. Exit")
    
    choice = input("\nChoose (1-3): ")
    
    if choice == '1':
        path = input("Enter image path: ")
        
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                currency, confidence = predict_image(img)
                print(f"\nResult: {currency}")
                print(f"Confidence: {confidence:.1%}")
                
                # Show image with result
                display = cv2.resize(img, (600, 400))
                cv2.putText(display, f"{currency}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display, f"{confidence:.1%}", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, "Press any key to close", (20, 350), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Result', display)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Cannot read image file")
        else:
            print("File not found")
    
    elif choice == '2':
        print("\nWebcam - Press 'q' to quit")
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            currency, confidence = predict_image(frame)
            
            cv2.putText(frame, currency, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"{confidence:.0%}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('Currency Recognition - Press Q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif choice == '3':
        print("Goodbye!")
        break
    
    else:
        print("Invalid choice")