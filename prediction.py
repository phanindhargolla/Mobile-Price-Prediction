from flask import Flask, render_template, request
from tensorflow.keras.models import model_from_json
import numpy as np
from Mobile import minmax

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            battery = int(request.form['battery'])
            fc = int(request.form['fc'])
            int_mem = int(request.form['int_mem'])
            weight = int(request.form['weight'])
            pc = int(request.form['pc'])
            res = int(request.form['res'])
            width = int(request.form['width'])
            ram = int(request.form['ram'])
            scheight = int(request.form['scheight'])
            scwidth = int(request.form['scwidth'])
            time = int(request.form['time'])
            blue = int(request.form['blue'])
            dual = int(request.form['dual'])
            gs = int(request.form['gs'])
            gs3 = int(request.form['gs3'])
            touch = int(request.form['touch'])
            wifi = int(request.form['wifi'])
            selected = [battery, fc, int_mem, weight, pc, res, width, ram, scheight, scwidth, time]
            cols = minmax.transform(np.array(selected).reshape(1, -1)).flatten().tolist()
            x = np.array(cols + [blue, dual, gs, gs3, touch, wifi]).reshape(1, -1)
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("model.h5")
            loaded_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            prediction_value = loaded_model.predict(x).argmax()
            return render_template('predict.html', prediction = prediction_value)
        except ValueError:
            return render_template('error.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1')