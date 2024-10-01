import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "kernel_size": 3,  # Size of the convolutional kernel
    "out_channels_1": 16,  # Number of filters in the first Conv1D layer
    "out_channels_2": 32,  # Number of filters in the second Conv1D layer
    "pool_size": 2,  # Size of the pooling window
    "fc1_size": 128,  # Number of units in the first fully connected layer
    "batch_size": 64,  # Batch size for training
    "lr": 0.0001,  # Learning rate
    "n_epochs": 100,  # Number of epochs for training
}


# Define a 1D CNN model using config
class CNNVelocityPredictor(nn.Module):
    def __init__(self, config):
        super(CNNVelocityPredictor, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=config["out_channels_1"],
            kernel_size=config["kernel_size"],
        )
        self.pool = nn.MaxPool1d(config["pool_size"])  # Max pooling layer
        self.conv2 = nn.Conv1d(
            in_channels=config["out_channels_1"],
            out_channels=config["out_channels_2"],
            kernel_size=config["kernel_size"],
        )
        self.fc1 = nn.Linear(
            config["out_channels_2"] * 48, config["fc1_size"]
        )  # Adjust input size after pooling
        self.fc2 = nn.Linear(config["fc1_size"], 1)  # Final output (velocity)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the saved data
slope_tensor, thickness_tensor, velocity_tensor = torch.load("glacier_data.pt")

# Prepare training data (stack slope and thickness as input features)
X = torch.stack((slope_tensor, thickness_tensor), dim=1)  # X shape: (n_samples, 2, 200)
y = velocity_tensor  # Target is the velocity

# Create a DataLoader for batching
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config["batch_size"], shuffle=True
)

# Initialize the CNN model, loss function, and optimizer
model = CNNVelocityPredictor(config).to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

# Training loop
for epoch in range(config["n_epochs"]):
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the average loss every epoch
    print(
        f"Epoch {epoch + 1}/{config['n_epochs']}, Loss: {running_loss / len(dataloader)}"
    )
