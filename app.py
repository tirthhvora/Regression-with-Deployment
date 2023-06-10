import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
#starting point from where app will be run

regression_model = pickle.load(open("regression_model.pkl", 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))
#loading the model

@app.route('/')
def home():
    return render_template('home.html')

#now we are going to make a predict API, where using postman or any other tool, we can send
#a request to our app, and then get the output

@app.route('/predict_api', methods = ['POST'])

def predict_api():
    data = request.json['data']
    #it means that whenever I hit this predict_api, the input will be in the json format that will be captured in the data key
    #after post request, whatever is in the 'data' we will capture it using request.json and it will get stored in the data variable
    print(data)
    #we have to do standardization here too
    #the data will be in key value pairs, hence i need to print only values
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regression_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)


