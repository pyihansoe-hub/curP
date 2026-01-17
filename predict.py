#!/usr/bin/env python3
"""
Currency Recognition - Prediction Module
Uses trained model to recognize currencies from images or webcam.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import joblib
import pickle

class CurrencyPredictor:
    """Handles currency prediction using trained model."""
    
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.model = None
        self.classes = []
        self.load_model()
    
    def load_model(self):
        """Load trained model and metadata."""
        try:
            model_path = self.model_dir / 'currency_model.joblib'
            classes_path = self.model_dir / 'classes.pkl'
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. Run train.py first."
                )
            
            self.model = joblib.load(model_path)
            with open(classes_path, 'rb') as f:
                self.classes = pickle.load(f)
            
            print(f"✅ Model loaded: {len(self.classes)} currencies")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def preprocess_image(self, image):
        """Preprocess image for prediction (same as training)."""
        image = cv2.resize(image, (100, 100))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = image.flatten() / 255.0
        return features
    
    def predict(self, image):
        """Predict currency from image."""
        try:
            features = self.preprocess_image(image)
            prediction_idx = self.model.predict([features])[0]
            
            # Get confidence
            probabilities = self.model.predict_proba([features])[0]
            confidence = probabilities[prediction_idx]
            
            # Safety check
            if prediction_idx >= len(self.classes):
                print(f"⚠️ Warning: Invalid prediction index {prediction_idx}")
                return "Unknown", 0.0
            
            return self.classes[prediction_idx], confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Error", 0.0
    
    def predict_from_file(self, image_path):
        """Predict currency from image file."""
        path = Path(image_path)
        if not path.exists():
            return f"File not found: {image_path}", 0.0
        
        image = cv2.imread(str(path))
        if image is None:
            return f"Cannot read image: {image_path}", 0.0
        
        return self.predict(image)
    
    def display_result(self, image, currency, confidence):
        """Display image with prediction results."""
        display = cv2.resize(image, (600, 400))
        
        # Color based on confidence
        if confidence > 0.7:
            color = (0, 255, 0)  # Green
        elif confidence > 0.5:
            color = (0, 200, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        # Add text
        cv2.putText(display, f"Currency: {currency}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display, f"Confidence: {confidence:.1%}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Add info
        cv2.putText(display, "Press any key to close", (20, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Currency Recognition', display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def webcam_mode(self):
        """Real-time prediction from webcam."""
        print("\n📷 Starting webcam...")
        print("Controls:")
        print("  Q - Quit")
        print("  S - Save current frame")
        print("  C - Capture & analyze")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        last_prediction = "N/A"
        last_confidence = 0.0
        last_update = 0
        save_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            
            # Update prediction every 0.5 seconds for performance
            if current_time - last_update > 0.5:
                currency, confidence = self.predict(frame)
                last_prediction = currency
                last_confidence = confidence
                last_update = current_time
            
            # Create display frame
            display = frame.copy()
            
            # Add prediction overlay
            if last_confidence > 0.5:
                color = (0, 255, 0) if last_confidence > 0.7 else (0, 200, 255)
                cv2.putText(display, last_prediction, (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(display, f"{last_confidence:.0%}", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(display, "Adjust position...", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.putText(display, "Point currency at camera", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            
            # Add controls info
            cv2.putText(display, "Q: Quit  |  S: Save  |  C: Capture", (20, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Currency Recognition - Live', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save frame
                filename = f"capture_{save_counter:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved: {filename}")
                save_counter += 1
            elif key == ord('c'):
                # Capture and analyze
                currency, confidence = self.predict(frame)
                print(f"\n📸 Capture: {currency} ({confidence:.1%})")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam stopped.")
    
    def run(self):
        """Main prediction interface."""
        print("="*50)
        print("💰 CURRENCY RECOGNITION SYSTEM")
        print("="*50)
        print(f"Available currencies: {', '.join(self.classes)}")
        
        while True:
            print("\n" + "="*30)
            print("MAIN MENU")
            print("="*30)
            print("1. Analyze image file")
            print("2. Real-time webcam")
            print("3. Test multiple images")
            print("4. Exit")
            
            try:
                choice = input("\nSelect option (1-4): ").strip()
                
                if choice == '1':
                    # File mode
                    path = input("Enter image path: ").strip()
                    currency, confidence = self.predict_from_file(path)
                    
                    if "not found" in currency.lower() or "cannot read" in currency.lower():
                        print(f"❌ {currency}")
                    else:
                        print(f"\n✅ Result: {currency}")
                        print(f"✅ Confidence: {confidence:.1%}")
                        
                        # Show image if prediction successful
                        if confidence > 0:
                            image = cv2.imread(path)
                            if image is not None:
                                self.display_result(image, currency, confidence)
                
                elif choice == '2':
                    # Webcam mode
                    self.webcam_mode()
                
                elif choice == '3':
                    # Batch test
                    folder = input("Enter folder path (or press Enter for current): ").strip()
                    if not folder:
                        folder = "."
                    
                    test_files = list(Path(folder).glob('*.jpg')) + \
                                list(Path(folder).glob('*.png')) + \
                                list(Path(folder).glob('*.jpeg'))
                    
                    if not test_files:
                        print("❌ No images found in folder")
                        continue
                    
                    print(f"\nTesting {len(test_files)} images...")
                    for img_path in test_files[:10]:  # Limit to 10 images
                        currency, confidence = self.predict_from_file(img_path)
                        print(f"  {img_path.name}: {currency} ({confidence:.1%})")
                
                elif choice == '4':
                    print("\n👋 Thank you for using Currency Recognition!")
                    break
                
                else:
                    print("❌ Invalid option. Please choose 1-4.")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Entry point for prediction script."""
    predictor = CurrencyPredictor()
    predictor.run()

if __name__ == "__main__":
    main()