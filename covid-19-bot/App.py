from flask import Flask, jsonify, redirect, url_for, request
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle as pickle 
import os


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/predict', methods=["POST"])
def predict():
    answers = request.json
    answers[0] = 1 if answers[0] == "female" else 2

    print(answers)
    tensor_answers = torch.tensor(answers, dtype=torch.float32)
    
    with open('/Users/daniel/Programming/Covid-19-bot/covid-19-bot/model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    print("Model loaded successfully!")
    loaded_model.eval()
    with torch.no_grad():
        predict = loaded_model(tensor_answers)
    print(predict)
    _, predicted_class = torch.max(predict, 0)
    # ### DATA LOADING (without creating a new StandardScaler)
    # testdata = pd.read_csv('Covid Data 2.csv').values
    # print(testdata.head())
    # unscaled_testdata = np.delete(testdata, [0,1], axis=1)
    # ## Note there is no scaling here
    check = {0:"no covid", 1:"mild", 2:"moderate", 3:"severe"}
    predicted_value = check[predicted_class.item()]
    
    # prediction = loaded_model.predict(unscaled_testdata)
    # print(prediction)
    print("I AM WORKING WELL!")
    print(predicted_value)

    return jsonify(predicted_value)


        

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "public", "data.csv")
    df = pd.read_csv(DATA_PATH)
    #print(df.head())

    df['DATE_DIED'] = [1 if each =="9999-99-99" else 0 for each in df["DATE_DIED"]]
    df['CLASIFFICATION_FINAL'] = [0 if each > 3 else each for each in df['CLASIFFICATION_FINAL']]
    print("HIIIIIII",df['CLASIFFICATION_FINAL'].value_counts())
    df.drop('USMER', axis=1, inplace=True)
    df.drop('MEDICAL_UNIT', axis=1, inplace=True)
    df.drop('PATIENT_TYPE', axis=1, inplace=True)
    #print(df.head())

    x = df.iloc[0:400, list(range(0,16)) + [17]]
    y = df['CLASIFFICATION_FINAL'].iloc[0:400]
    x.to_numpy()
    y.to_numpy()
    x_test = df.iloc[400:800, list(range(0,16)) + [17]]
    y_test = df['CLASIFFICATION_FINAL'].iloc[400:800]

    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    scaler.fit(x_test)
    x_test_scaled = scaler.transform(x_test)

    class MultiLayerLogisticRegression(nn.Module):
        def __init__(self, input_size, num_classes):
            super(MultiLayerLogisticRegression, self).__init__()
            self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    app.run(debug=False, host='0.0.0.0', port=8000)