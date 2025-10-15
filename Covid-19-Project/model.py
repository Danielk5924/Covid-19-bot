#Train and Validate the model
import torch
import copy
from copy import deepcopy
from sklearn.model_selection import KFold
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

#This file contains the origional code used to train and test the model


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
          nn.Linear(input_size, 32),
          nn.ReLU(),
          # nn.Dropout(p=0.3),
          nn.Linear(32, 64),
          nn.ReLU(),
          # nn.Dropout(p=0.3),
          nn.Linear(64, 32),
          nn.ReLU(),
          # nn.Dropout(p=0.3),
          nn.Linear(32, num_classes)
        )

    def forward(self, x):
      logits = self.linear_relu_stack(x)
      return logits
      #return

def train_one_epoch(x):
    cumulative_loss = 0
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(x):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.float()
        #print(inputs)
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report




dataset = train_dataset
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Store fold metrics
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"\n--- Fold {fold + 1} ---")

    # Dataloaders for this fold
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64)

    # Model, optimizer, loss, scheduler (per fold)
    model = MultiLayerLogisticRegression(input_size=17, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    for epoch in range(1, 10):
        model.train()
        train_one_epoch(train_loader)

        # Validation within fold
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch.float())
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        scheduler.step(val_loss)

        print(f"Epoch {epoch:2d} - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Store performance for this fold
    fold_results.append((val_loss, val_accuracy))

avg_loss = sum([x[0] for x in fold_results]) / len(fold_results)
avg_acc = sum([x[1] for x in fold_results]) / len(fold_results)

print(f"\nAverage Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.2f}%")





# model = MultiLayerLogisticRegression(input_size=17, num_classes=4)

# class_weights = torch.tensor([1.0, 2.0, 1.5, 1.0])  # Example
# criterion = nn.CrossEntropyLoss(weight=class_weights)
# #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
# #optimizer = torch.optim.Adam(model.parameters(), lr=0.001
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)




# loss_array = []
# def train(epochs):
#   model.train()
#   global loss_array
#   for epoch in range(epochs):
#     loss = train_one_epoch()
#     loss_array.append(loss)
#     print(loss)

#     # Make sure gradient tracking is on, and do a pass over the data

# with torch.enable_grad():
#   train(1000)

best_model = copy.deepcopy(model)


best_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.float()
        outputs = best_model(inputs)

        # takes the most likely of all the different predicted labels
        _, predicted = torch.max(outputs, dim=1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"FINAL Accuracy: {accuracy:.2f}%")


#epoch: 1500 and accuracy: 62
