from flask import Flask, redirect, url_for, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle as pickle 

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/predict', methods = ["POST"])
def predict():
    answers = request.json
    print(answers)

    with open('finalizedmode.sav', 'rb') as f:
        loaded_model = pickle.load(f)

    ### DATA LOADING (without creating a new StandardScaler)
    testdata = pd.read_csv('Covid Data 2.csv').values
    print(testdata.head())
    unscaled_testdata = np.delete(testdata, [0,1], axis=1)
    ## Note there is no scaling here

    prediction = loaded_model.predict(unscaled_testdata)
    print(prediction)
    return prediction.tolist()


        

if __name__ == '__main__':
    testdata = pd.read_csv('Covid Data 2.csv')
    #print(testdata.head())
    app.run(debug=True, host='0.0.0.0', port=8080)