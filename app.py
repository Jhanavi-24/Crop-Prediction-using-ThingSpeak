from flask import Flask, render_template, request
import requests
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# import json
import pandas as pd
import joblib

app = Flask(__name__)
app.url_map.strict_slashes = False

# API data
channel_id = '562742'
read_key = 'B7KXZ1OS1873O8ET'
num_entries = 1

# Loading model and labelencoder
model = joblib.load('crop_pred_model.pkl')
le = joblib.load('crop_pred_labelencoder.pkl')

# Function to retrieve data from API
def get_data_from_thingspeak(channel_id, read_key, num_entries):
    url = f'https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={read_key}&results={num_entries}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # Extracting relevant data from the response
        entries = data['feeds']
        field1_data = [entry['field1'] for entry in entries]
        field2_data = [entry['field2'] for entry in entries]
        field3_data = [entry['field3'] for entry in entries]
        field4_data = [entry['field4'] for entry in entries]
        return field1_data, field2_data, field3_data, field4_data
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

# Predict function
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        target_values = []
        target_values.append(get_data_from_thingspeak(channel_id, read_key, num_entries))
        print(target_values)
        data = {
            'temperature': target_values[0][1],
            'humidity': target_values[0][2],
            'ph': target_values[0][0],
            'rainfall': target_values[0][3]
        }
        test_data = pd.DataFrame(data)
        print(test_data)
        result = model.predict(test_data)
        result = le.inverse_transform(result)
        print(result)
        return render_template('index.html', result=result, ph = data['ph'], temp = data['temperature'], hum = data['humidity'], rain = data['rainfall'])

if __name__ == "__main__":
    app.run(debug=True)