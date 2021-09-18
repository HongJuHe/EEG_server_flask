from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import struct
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
    return render_template('graph.html')

#upload html rendering
@app.route('/upload')
def render_file():
    return render_template('send_file.html')

#file upload work
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    temp_list_f = []

    if request.method == 'POST':
        f = request.files['file'].read()
        #f.save('/home/ubuntu/my_tensorflow/uploads/'+secure_filename(f.filename))
        #temp_list = np.loadtxt(f, dtype='float')
        temp_string = f.decode()
        temp_list = np.array(map(float, temp_string.split()))
        print(temp_list)

        for i in temp_list:
            if i == '-':
                check = True
            elif i == 'end':
                flag = False
            else:
                if check:
                    check = False
                    temp_list_f.append(float(i)*-1)
                else:
                    temp_list_f.append(float(i))
        print(temp_list_f)

        return render_template('send_result.html')

if __name__ == '__main__':
    app.debug = True
    HOST = '172.31.29.204'
    PORT = 8080
    app.run(host=HOST, port=PORT)