from flask import Flask, request
import cv2
import numpy as np
import joblib
import pickle
import os

app = Flask(__name__)

model = joblib.load('models/currency_model.joblib')
with open('models/classes.pkl', 'rb') as f:
    classes = pickle.load(f)

def predict(img):
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten() / 255.0
    pred = model.predict([img])[0]
    prob = model.predict_proba([img])[0][pred]
    return classes[pred], prob

@app.route('/')
def index():
    return '''
    <h2>Currency Recognition</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file"><br>
        <input type="submit" value="Upload">
    </form>
    <p>Currencies: ''' + ', '.join(classes) + '''</p>
    '''

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    
    currency, conf = predict(img)
    
    return f'''
    <h3>Result: {currency}</h3>
    <p>Confidence: {conf:.1%}</p>
    <a href="/">Back</a>
    '''

if __name__ == '__main__':
    app.run()