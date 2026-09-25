import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# --- 1. Dummy Time Series Data Creation (Simulating Stock Prices) ---
# Generate a simple sinusoidal time series
def generate_time_series(num_points, freq=0.1, amplitude=1.0, noise_std=0.05):
    time = np.arange(num_points)
    series = amplitude * np.sin(freq * time) + np.random.normal(0, noise_std, num_points)
    return series

num_data_points = 1000
raw_series = generate_time_series(num_data_points)

# Prepare data for sequence prediction (input sequence -> next value)
sequence_length = 10 # Number of past points to look at for prediction
X, Y = [], []
for i in range(len(raw_series) - sequence_length):
    X.append(raw_series[i:i + sequence_length])
    Y.append(raw_series[i + sequence_length])

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) # Add feature dimension: (samples, seq_len, 1)
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1) # Add output dimension: (samples, 1)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(X, Y)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 2. Model Hyperparameters ---
input_size = 1      # Each time step has 1 feature (price)
hidden_size = 64    # Dimension of LSTM hidden state
num_layers = 1      # Number of LSTM layers
output_size = 1     # Predicting 1 future value
learning_rate = 0.005
num_epochs = 50

# --- 3. Define the Time Series Predictor Model ---
class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TimeSeriesPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # Blank 1: nn.LSTM layer
        self.fc = nn.Linear(hidden_size, output_size)   # Blank 2: nn.Linear layer

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass input through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0)) # Blank 3: Call self.lstm with x and (h0, c0)

        # Use the output from the last time step for prediction
        # out shape: (batch_size, sequence_length, hidden_size)
        # We want out[:, -1, :] which is (batch_size, hidden_size)
        prediction = self.fc(out[:, -1, :]) # prediction shape: (batch_size, output_size)
        
        return prediction

# --- 4. Instantiate Model, Loss, and Optimizer ---
model = TimeSeriesPredictor(input_size, hidden_size, num_layers, output_size)
print(f"\nModel Architecture:\n{model}")

criterion = nn.MSELoss() # Blank 4: nn.MSELoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 5. Training Loop ---
print("\n--- Starting Training ---")
for epoch in range(num_epochs):
    model.train() # Set the model to training mode
    total_loss = 0

    for i, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

print("--- Training Finished ---")

# --- 6. Simple Evaluation ---
print("\n--- Evaluating Model ---")
model.eval() # Set the model to evaluation mode

with torch.no_grad(): # Disable gradient calculations
    all_inputs, all_targets = dataset[:] # Get entire dataset
    
    predictions = model(all_inputs)
    
    # Calculate overall MSE on the dataset
    eval_loss = criterion(predictions, all_targets).item()
    
    print(f'Evaluation MSE on the {all_targets.size(0)} data points: {eval_loss:.4f}')