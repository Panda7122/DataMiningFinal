import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
_random_state = 42
# Load the dataset
data = pd.read_csv('dataset/Mental Health Data/processed_train.csv')
testdata = pd.read_csv('dataset/Mental Health Data/processed_test.csv')

# Separate the features and the target variable
X = data.drop('Depression', axis=1)
y = data['Depression']
# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=_random_state)
# val_X, X_test, val_Y, y_test = train_test_split(X_test, y_test, test_size = 0.6, random_state = _random_state)

import torch.nn as nn
import torch.optim as optim

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
test_X_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
test_Y_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(test_X_tensor, test_Y_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize the model, loss function and optimizer
input_dim = X_train.shape[1]
model = SimpleNN(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
TrainLoss = []
ValidLoss = []
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    TrainLoss.append(running_loss/len(train_loader))
    ValidLoss.append(val_loss/len(test_loader))
    print(f'Epoch {epoch+1}/{num_epochs}| Training Loss: {running_loss/len(train_loader)}| Validation Loss: {val_loss/len(test_loader)}')
import matplotlib.pyplot as plt

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(TrainLoss, label='Training Loss')
plt.plot(ValidLoss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('NNLossPlot.png')
# Evaluate the model with ROC Curve
model.eval()
y_true = []
y_scores = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        y_true.extend(labels.cpu().numpy())
        y_scores.extend(outputs.cpu().numpy())

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('NNROC.png')

y_scores = np.sign(y_scores)
# print(y_scores)
accuracy = accuracy_score(y_true, y_scores)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_true, y_scores))
print('Confusion Matrix:')
print(confusion_matrix(y_true, y_scores))

X_pred = torch.tensor(testdata.values, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    y_ans = model(X_pred)
    y_ans = (y_ans > 0.5).float()

testdata = pd.read_csv('dataset/Mental Health Data/test.csv')

with open("NNOutput.csv", 'w') as f:
    f.write('id,Depression\n')
    for i, idx in enumerate(testdata['id']):
        f.write(f'{idx},{int(y_ans[i].item())}\n')