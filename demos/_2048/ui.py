from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
import numpy as np
from .model import CNN
import os


app = Flask(__name__)
net = CNN()
pre_arrs = None
file_dir_path = os.path.dirname(__file__)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/record", methods=['POST'])
def move():
    message = request.form
    key = message['key']
    h = message['height']
    w = message['width']
    arrs = message['array'][1:-1]
    with open(f'{file_dir_path}/train_{h}_{w}.txt', 'a') as f1:
        f1.write(arrs)
        f1.write('\n')

    with open(f'{file_dir_path}/label_{h}_{w}.txt', 'a') as f1:
        f1.write(key)
        f1.write('\n')

    return jsonify({'status': 'OK'})


@app.route("/train", methods=['POST'])
def train():
    message = request.form
    h = int(message['height'])
    w = int(message['width'])
    x = []
    y = []
    with open(f'{file_dir_path}/train_{h}_{w}.txt', 'r') as f1:
        for line in f1.readlines():
            line = list(map(int, line.strip().split(',')))
            x.append(line)

    with open(f'{file_dir_path}/label_{h}_{w}.txt', 'r') as f2:
        for target in f2.readlines():
            y.append(int(target))

    net.training(x, y, h, w)
    return jsonify({'status': 'OK'})


@app.route("/ai", methods=['POST'])
def ai():
    global pre_arrs
    message = request.form
    arrs = list(map(int, message['array'][1:-1].split(',')))
    h = int(message['height'])
    w = int(message['width'])

    if pre_arrs is not None and np.equal(arrs, pre_arrs).all():
        key = int(np.random.choice([0, 1, 2, 3]))
    else:
        key = net.prediction(arrs, h, w)[0]

    pre_arrs = arrs

    return jsonify({'status': 'OK', 'key': key})


def go():
    app.run(debug=True)


