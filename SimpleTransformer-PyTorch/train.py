import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleTransformer
from dataset import generate_data
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# optional: plot the training loss and save to a file
plot_training_loss = False

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
embed_size = 512
sequence_length = 3
vocab_size = 10  # Simple vocab size for demonstration
num_samples = 10000
batch_size = 256
learning_rate = 0.001
epochs = 100

# Model, loss, and optimizer
model = SimpleTransformer(embed_size, sequence_length, vocab_size).to(device)
loss_fn = nn.CrossEntropyLoss()

# Optionally introduce weight decay
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Generating data
inputs, targets = generate_data(num_samples, sequence_length, vocab_size)

# Create a TensorDataset to hold the inputs and targets
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

losses_per_epoch = []

# Optionally introduce gradient clipping
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for input_batch, target_batch in dataloader:
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)

        optimizer.zero_grad()
        output = model(input_batch)
        loss = loss_fn(output.view(-1, vocab_size), target_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {loss:.6f}")
    losses_per_epoch.append(loss)

# Example sequence
sample_sequence = [1, 2, 3]
sample_tensor = (
    torch.tensor(sample_sequence, dtype=torch.long).unsqueeze(0).to(device)
)  # Add batch dimension and send to device

model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient computation for inference
    predictions = model(sample_tensor)
    predicted_index = predictions.argmax(
        -1
    )  # Get the index of the max log-probability for the last position

predicted_number = predicted_index[0, -1].item()  # Convert to Python number
print(f"Input Sequence: {sample_sequence}")
print(f"Predicted Next Number: {predicted_number}")

if plot_training_loss:
    plt.plot(losses_per_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("training_loss.png")
