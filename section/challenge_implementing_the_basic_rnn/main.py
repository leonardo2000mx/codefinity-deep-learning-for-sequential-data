import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# --- 1. Dummy Dataset Creation ---
# Parameters for dummy data
num_samples = 200
sequence_length = 10
input_size = 1 # Each element in sequence is a single number
num_classes = 2 # Binary classification (e.g., sum > threshold or not)

# Generate random sequences
# Sequences will be between 0 and 1
sequences_np = np.random.rand(num_samples, sequence_length, input_size).astype(np.float32)

# Create labels based on a simple rule (e.g., sum of sequence > threshold)
threshold = sequence_length * 0.5 # Example threshold
labels_np = (sequences_np.sum(axis=(1, 2)) > threshold).astype(np.int64)

# Convert to PyTorch tensors
sequences = torch.from_numpy(sequences_np)
labels = torch.from_numpy(labels_np)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(sequences, labels)
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 2. Model Hyperparameters ---
hidden_size = 32
num_layers = 1
learning_rate = 0.005
num_epochs = 50

# --- 3. Define the RNN Model ---
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = self.hidden_size # Blank 1
        self.fc =  self.num_layers # Blank 2

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, hn = self.rnn(x, h0)
        out = self.fc(hn[-1])
        
        return out

# --- 4. Instantiate Model, Loss, and Optimizer ---
model = SimpleRNN(input_size, hidden_size, num_layers, num_classes)
print(f"\nModel Architecture:\n{model}")

criterion = ___ # Blank 3
optimizer = ___ # Blank 4

# --- 5. Training Loop ---
print("\n--- Starting Training ---")
for epoch in range(num_epochs):
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    model.train() # Set the model to training mode

    for i, (sequences, labels) in enumerate(dataloader):
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted_classes = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted_classes == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    epoch_accuracy = 100 * correct_predictions / total_samples

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

print("--- Training Finished ---")

# --- 6. Simple Evaluation ---
print("\n--- Evaluating Model ---")
model.eval() # Set the model to evaluation mode

with torch.no_grad(): # Disable gradient calculations
    all_sequences, all_labels = dataset[:] # Get entire dataset
    
    outputs = model(all_sequences)
    _, predicted = torch.max(outputs.data, 1)
    
    total = all_labels.size(0)
    correct = (predicted == all_labels).sum().item()
    
    print(f'Accuracy of the model on the {total} training sequences: {100 * correct / total:.2f} %')
