from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import struct
import os
import numpy as np
from numpy.lib.npyio import save
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.models import load_model
import json
import datetime as dt
import boto3

app = Flask(__name__)

#main page
@app.route('/')
def main():
    return render_template('index.html')

#map page
@app.route('/map')
def map():
    return render_template('map.html')

#records page
@app.route('/graph')
def record():
    send_data = []
    #get depression_result
    depression_data = []
    temp_data = {}
    with open('./static/data/depression_result.json', 'r') as f:
        temp_data = json.load(f)
    
    temp_d = temp_data['depression']
    for d in temp_d:
        d_data = []
        d_data.append(d['date'])
        d_data.append(d['result'])
        #d_data.append(temp_data[d]['result'])
        d_data.append(d['percent'])
        depression_data.append(d_data)
    
    #print(depression_data)
    
    #get eeg data
    eeg_data = []
    file_name = depression_data[-1][0] + ".txt"
    f = open(os.path.join('./static/data/', file_name), "r")
    content = f.read()
    content_list = content.split(" ")
    f.close()

    for i in content_list:
        if i != '':
            eeg_data.append(float(i))

    #print(eeg_data[0])
    #print(type(eeg_data))

    #get intent
    intent = []
    dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2')
    table = dynamodb.Table('user_db2')
    response = table.scan()
    items = response['Items']
    for item in items:
        temp_intent = []
        for e in item['events']:
            if "input_channel" in e:
                if e['input_channel'] == 'socketio':
                    #print(e['parse_data']['intent_ranking'])
                    n = e['parse_data']['intent_ranking'][0]
                    if n['confidence'] > 0.8:
                        temp_intent.append(n['name'])

        if (len(temp_intent) != 0):
            intent.append(temp_intent)

    send_data.append(depression_data)
    send_data.append(eeg_data[:1000])
    send_data.append(intent)

    #print(send_data)
    return render_template('graph.html', value=send_data)

#upload html rendering
@app.route('/eeg')
def render_file():
    return render_template('eeg.html')

#file upload work
@app.route('/fileUpload', methods = ['POST'])
def upload_file():

    if request.method == 'POST':
        f = request.files['file']
        #f_name = secure_filename(f.filename)
        x = dt.datetime.now()
        f_name = str(x.year)+"_"+str(x.month)+"_"+str(x.day)+".txt" 
        path_dir = '/home/ubuntu/webpage/static/data/'
        
        f.save(os.path.join(path_dir, f_name))
        print(f_name)
        #f.save('/home/ubuntu/my_tensorflow/uploads/'+secure_filename(f.filename))
        #temp_list = np.loadtxt(f, dtype='float')
        #temp_string = f.decode()
        #temp_list = np.array(map(float, temp_string.split()))
        #print(temp_list)

        f = open(os.path.join(path_dir, f_name), "r")
        content = f.read()
        content_list = content.split(" ")
        f.close()
        #print(len(content_list))
        temp_list = []
        for i in content_list:
            if i != '':
                temp_list.append(float(i))
        
        print(len(temp_list))
        data_arr = np.array(temp_list[:210000])
        print(np.shape(data_arr))
        data_arr = np.reshape(data_arr, (1, 500, 420))
        print(np.shape(data_arr))
        model = load_model('../server/eeg_deeplearning_conv1d_lstm.h5')
        result = model.predict(data_arr)
        result = result.tolist()
        print(result)
        result = result[0]

        #json으로 결과 저장
        threshold = 0.4
        send_data = []
        if result[0] >= threshold:
            send_data.append(0)
            send_data.append(result[0])
        else:
            send_data.append(1)
            send_data.append(result[1])
        
        #send_data = [0, 0.9159774780273438]

        temp_data = {}
        with open('./static/data/depression_result.json', 'r') as f:
            temp_data = json.load(f)

        temp_date = str(x.year) + "_" + str(x.month) + "_" + str(x.day)
        temp_data["depression"].append({
            "date": temp_date,
            "result": send_data[0],
            "percent": send_data[1]
        })

        with open('./static/data/depression_result.json', 'w') as f:
            json.dump(temp_data, f, indent="\t")

        return render_template('send_result.html', value=send_data)

if __name__ == '__main__':
    app.debug = True
    HOST = '172.31.29.204'
    PORT = 8080
    #app.run(host=HOST, port=PORT)
    app.run(host=HOST, port=PORT)