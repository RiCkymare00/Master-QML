import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import *

# Define model
class SimpleMLP(nn.Module):   #MLP = multi layer perception
    def __int__(self, input_size = 28*28,hidden_size=128):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

transform = transform.Compose([
    transform.ToTensor(),
    transform.Normaliza((0.1307),(0.3081))
])

full_dataset = dataset.MNIST(root='./data',train=True, transform=transform, download = True)
test_dataset = dataset.MNIST(root='./data',train=False, transform=transform, download = True)

train_size = int(0.7*len(full_dataset))
eval_size = int(0.1*len(full_dataset))
test_size = len(full_dataset)

train_dataset, test_dataset, eval_dataset = random_split()
train_loader = DataLoader()
test_loader = DataLoader()
eval_loader = DataLoader()

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Training loop
num_epochs = 10
train_losses = []
eval_losses = []

for epoch in range(num_epochs):
    model.train()
    total_los = 0

    for images, labels in train_loader:
        images,labels = images.to(device), label.float().to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

avg_train_loss = total_los/len(train_loader)
train_losses.append(avg_train_loss)

model.eval()
eval_loss = 0
with torch.no_grad():
    for images, labels in eval_loader:
        images,labels = images.to(device), label.float().to(device)
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        eval_loss += loss.item()

avg_eval_loss = eval_loss/len(eval_loader)
eval_losses.append(avg_eval_loss)

# ...