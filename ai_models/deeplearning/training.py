import torch
import torch.nn as nn
import torch.optim as optim
from network import SimpleModel


model = SimpleModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate random training data
x_train = torch.randn(100, 10)  # 100 samples, 10 features each
y_train = torch.randn(100, 1)  # 100 target values

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model checkpoint with .ckpt extension
torch.save(model.state_dict(), "model_checkpoint.ckpt")

print("Model training complete and checkpoint saved.")
