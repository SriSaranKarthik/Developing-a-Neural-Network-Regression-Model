# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Design a neural network model to solve a regression problem using a single input feature. The model uses multiple fully connected layers with ReLU activation to predict a continuous output value. Train the network using a loss function and optimizer over several epochs to minimize error. Track and display the training loss during the learning process.

## Neural Network Model
Include the neural network model diagram.


<img width="1820" height="1017" alt="Screenshot 2026-02-02 094607EXP1" src="https://github.com/user-attachments/assets/91177b10-6ef2-428c-b60f-6fbbf3c926d2" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: K.SriSaran Karthik

### Register Number: 212224230275

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('SampleSheet.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name: K.SriSaran Karthik
# Register Number:212224230275
class NeuralNet(nn.Module):
  def init(self):
        super().init()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss':[]}
  
  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.001)

# Name: K.SriSaran Karthik
# Register Number:212224230275
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```

### Dataset Information
<img width="140" height="377" alt="image" src="https://github.com/user-attachments/assets/0ce489ca-2c75-4f0c-b514-8a5e29e45e17" />


### OUTPUT
<img width="295" height="178" alt="image" src="https://github.com/user-attachments/assets/0931c72d-d534-46d6-b99a-93d6bfa6f703" />
<img width="180" height="28" alt="image" src="https://github.com/user-attachments/assets/48e31b15-f061-405f-a795-27520477a29a" />

### Training Loss Vs Iteration Plot
<img width="644" height="462" alt="image" src="https://github.com/user-attachments/assets/8e3ad282-8617-49d5-8311-c1d1771c579d" />



### New Sample Data Prediction
<img width="233" height="22" alt="image" src="https://github.com/user-attachments/assets/9994c91c-bfe5-4851-a6d1-46cdf00b07a6" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
