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
from torch.utils.data import DataLoader, TensorDataset



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
    print(df.head())

    
x = df.iloc[0:4000, list(range(0,21))]
y = df['Severity'].iloc[0:4000]


x.to_numpy()
y.to_numpy()
x_test = df.iloc[4000:8000, list(range(0,21))]
y_test = df['Severity'].iloc[4000:8000]

scaler = StandardScaler()
scaler.fit(x)
scaler.fit(x_test)
x_scaled = scaler.transform(x)
x_test_scaled = scaler.transform(x_test)

x_Traintensor = torch.tensor(x_scaled)
y_Traintensor = torch.tensor(y.values)
x_Testtensor = torch.tensor(x_test_scaled)
y_Testtensor = torch.tensor(y_test.values)


train_dataset = TensorDataset(x_Traintensor, y_Traintensor)
test_dataset = TensorDataset(x_Testtensor, y_Testtensor)

#Test differnt batch sizes with cross validaiton
test_loader = DataLoader(test_dataset, shuffle=True, batch_size = 96)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size = 96)

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