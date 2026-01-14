import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Transformation for images
transform = transforms.Compose([transforms.ToTensor()])

# Use FakeData dataset
train_dataset = datasets.FakeData(
    size=1000,
    image_size=(1, 28, 28),
    num_classes=10,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Simple CNN model with dynamic flatten size
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        
        # Compute flatten size dynamically
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, 28, 28)
            sample_output = self.conv1(sample_input)
            self.flatten_size = sample_output.numel()  # total elements
        
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.flatten(x, 1)  # flatten except batch dimension
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, optimizer, and loss
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop (1 epoch)
for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % 10 == 0:  # Print every 10 batches
        print(f"Batch {batch_idx}, Loss: {loss.item()}")


