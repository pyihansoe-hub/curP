#!/usr/bin/env python3
"""
Currency Recognition - Training Module
Trains a model on currency images.
"""

import os
import sys
import traceback
from pathlib import Path
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
import pickle

class CurrencyTrainer:
    """Handles training of currency recognition model."""
    
    def __init__(self, data_dir='datasets', model_dir='models'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
    def validate_dataset(self):
        """Check if dataset structure is valid."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory '{self.data_dir}' not found.")
        
        folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
        if len(folders) < 2:
            raise ValueError(f"Need at least 2 currency folders. Found: {len(folders)}")
        
        return sorted([f.name for f in folders])
    
    def load_and_process_images(self, folders):
        """Load images and extract features."""
        X, y = [], []
        class_names = []
        
        print("📷 Loading images...")
        for label, folder in enumerate(folders):
            class_names.append(folder)
            folder_path = self.data_dir / folder
            count = 0
            
            # Get all image files
            image_files = list(folder_path.glob('*.jpg')) + \
                         list(folder_path.glob('*.png')) + \
                         list(folder_path.glob('*.jpeg'))
            
            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"  ⚠️ Skipping unreadable: {img_path.name}")
                        continue
                    
                    # Preprocess
                    img = cv2.resize(img, (100, 100))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = img.flatten() / 255.0
                    
                    X.append(img)
                    y.append(label)
                    count += 1
                    
                except Exception as e:
                    print(f"  ❌ Error processing {img_path.name}: {e}")
            
            print(f"  ✅ {folder}: {count} images")
            
        if not X:
            raise ValueError("No valid images found in dataset!")
        
        return np.array(X), np.array(y), class_names
    
    def train(self, X, y, class_names):
        """Train the SVM model."""
        print("\n🤖 Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        # Train model
        model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        print(f"   Training accuracy: {train_acc:.2%}")
        print(f"   Testing accuracy: {test_acc:.2%}")
        
        if train_acc > 0.95 and test_acc < 0.7:
            print("   ⚠️  Warning: Possible overfitting detected!")
        
        return model
    
    def save_model(self, model, class_names):
        """Save trained model and metadata."""
        # Save model
        model_path = self.model_dir / 'currency_model.joblib'
        joblib.dump(model, model_path)
        
        # Save class names
        classes_path = self.model_dir / 'classes.pkl'
        with open(classes_path, 'wb') as f:
            pickle.dump(class_names, f)
        
        # Save metadata
        metadata = {
            'feature_size': model.n_features_in_,
            'num_classes': len(class_names),
            'classes': class_names,
            'accuracy': model.score  # Reference to score method
        }
        
        meta_path = self.model_dir / 'metadata.pkl'
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        return model_path
    
    def run(self):
        """Main training pipeline."""
        print("="*50)
        print("💰 CURRENCY RECOGNITION TRAINER")
        print("="*50)
        
        try:
            # Step 1: Validate dataset
            folders = self.validate_dataset()
            print(f"📁 Found {len(folders)} currency types:")
            for folder in folders:
                print(f"   • {folder}")
            
            # Step 2: Load and process images
            X, y, class_names = self.load_and_process_images(folders)
            print(f"\n📊 Dataset summary:")
            print(f"   Total images: {len(X)}")
            print(f"   Feature size: {X.shape[1]}")
            
            # Step 3: Train model
            model = self.train(X, y, class_names)
            
            # Step 4: Save model
            model_path = self.save_model(model, class_names)
            
            print("\n" + "="*50)
            print("✅ TRAINING COMPLETE!")
            print("="*50)
            print(f"Model saved to: {model_path}")
            print(f"Currencies recognized: {', '.join(class_names)}")
            print(f"\nTo use: python predict.py")
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            traceback.print_exc()
            sys.exit(1)

def main():
    """Entry point for training script."""
    trainer = CurrencyTrainer()
    trainer.run()

if __name__ == "__main__":
    main()