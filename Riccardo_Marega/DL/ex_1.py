import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import torchsummary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define model
class SimpleMLP(nn.Module):   # MLP = Multi-Layer Perceptron
    def __init__(self, input_size=28*28, hidden_size=128):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Binary output
        self.sigmoid = nn.Sigmoid()  # Sigmoid for probability output
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Convert labels: Even -> 0, Odd -> 1
full_dataset.targets = (full_dataset.targets % 2 == 1).long()
test_dataset.targets = (test_dataset.targets % 2 == 1).long()

train_size = int(0.7 * len(full_dataset))
eval_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - eval_size

train_dataset, eval_dataset, test_dataset = random_split(full_dataset, [train_size, eval_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=True)

print(train_size)
print(test_size)
print(eval_size)

# Initialize the model
model = SimpleMLP().to(device)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
train_losses = []
eval_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0  

    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()  # Error back-propagation (that's the real training phase)
        optimizer.step()  # Update the weights according to the gradient
        total_loss += loss.item()  

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Evaluation
    model.eval()    
    eval_loss = 0
    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze()  # Forward pass
            loss = criterion(outputs, labels)
            eval_loss += loss.item()

    avg_eval_loss = eval_loss / len(eval_loader)
    eval_losses.append(avg_eval_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] -> Train Loss: {avg_train_loss:.4f} | Eval loss: {avg_eval_loss:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), eval_losses, label="Eval Loss", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss")
plt.legend()
plt.show()

# Test and train accuracy
model.eval()
correct_train = 0
total_train = 0
correct_test = 0
total_test = 0

# Train accuracy
with torch.no_grad():
    for images, labels in train_loader:  
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).squeeze()
        predicted = (outputs > 0.5).long()
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

train_acc = 100 * correct_train / total_train

# Test accuracy
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).squeeze()
        predicted = (outputs > 0.5).long()
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)

test_acc = 100 * correct_test / total_test

print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")  