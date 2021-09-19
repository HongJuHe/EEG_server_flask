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
@app.route('/statistics')
def record():
    #통계 호출
    return render_template('graph.html')

#upload html rendering
@app.route('/upload')
def render_file():
    return render_template('eeg.html')

#file upload work
@app.route('/fileUpload', methods = ['POST'])
def upload_file():

    if request.method == 'POST':
        f = request.files['file']
        #f_name = secure_filename(f.filename)
        x = dt.datetime.now()
        f_name = str(x.year)+"_"+str(x.month)+"_"+str(x.day)+".json" 
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
        #json으로 결과 저장
        threshold = 0.4
        send_data = []
        if result[0][1] >= threshold:
            send_data.append(1)
            send_data.append(result[0][1])
        else:
            send_data.append(0)
            send_data.append(result[0][0])
        
        #send_data = [0, 0.9159774780273438]

        big_data = {}

        with open('./static/data/depression_result.json', 'r') as f:
            big_data = json.load(f)

        data = {}
        data['result'] = send_data[0]
        data['percent'] = send_data[1]
        big_data[str(x.year)+"_"+str(x.month)+"_"+str(x.day)] = data
        
        with open('./static/data/depression_result.json', 'w') as outfile:
            json.dump(big_data, outfile, indent="\t")

        return render_template('send_result.html', value=send_data)

if __name__ == '__main__':
    app.debug = True
    HOST = '172.31.29.204'
    PORT = 8080
    app.run(host=HOST, port=PORT)