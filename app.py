import pickle
from flask import Flask, request, url_for, render_template
from flask import app, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('regressionModel.pkl', 'rb'))
scalar = pickle.load(open('scalar.pkl','rb'))
@app.route('/')
def Home():
    return render_template('home.html')

@app.route('/predict', methods=['POST']) # predict api
def predict_api():
    data= request.json()
    print(data)
    #data is in a Ke value reshape for predication
    print(np.array(list(data.values())).reshape(1, -1))
    data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predicat(data)
    return jsonify(output[0])

if __name__ == '__main__':
    app.run(debug=True)
    
    