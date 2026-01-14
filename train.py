import os
import cv2
import numpy as np
import joblib
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split

folders = [f for f in os.listdir('datasets') if os.path.isdir(f'datasets/{f}')]
folders.sort()

X = []
y = []
class_names = []

for label, folder in enumerate(folders):
    class_names.append(folder)
    folder_path = f'datasets/{folder}'
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.resize(img, (100, 100))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.flatten() / 255.0
                X.append(img)
                y.append(label)

if len(folders) < 2:
    print("Need at least 2 currency folders")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/currency_model.joblib')
with open('models/classes.pkl', 'wb') as f:
    pickle.dump(class_names, f)

print(f"Model saved")
print(f"Currencies: {class_names}")