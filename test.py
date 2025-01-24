# import requests
# import time
# import random

# while True:
#     print("Sending...", end="\r")
#     # data1 = random.randint(10,50)
#     # data2 = random.randint(50,100)
#     URL = f'https://api.thingspeak.com/update?api_key=6MV4KNRI8P8JUP02&field1=0&field1={data1}&field2={data2}'
#     response = requests.get(URL)
#     print('The response code is: ', response)
#     print('Data sent', data1, data2)
#     time.sleep(15)

import requests
import json
import pandas as pd
import joblib

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

model = joblib.load('crop_pred_model.pkl')
le = joblib.load('crop_pred_labelencoder.pkl')

channel_id = '562742'
read_key = 'B7KXZ1OS1873O8ET'
num_entries = 1  # Number of entries to fetch
target_values = []
target_values.append(get_data_from_thingspeak(channel_id, read_key, num_entries))
print(target_values)

data = {
    'temperature': target_values[0][1],
    'humidity': target_values[0][2],
    'ph': target_values[0][0],
    'rainfall': target_values[0][3]
}
print(data)
test_data = pd.DataFrame(data)
print(test_data)
test_data = pd.DataFrame(data)
print(test_data)
result = model.predict(test_data)
result = le.inverse_transform(result)
print(result)