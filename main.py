import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),                        # converts image to tensor (0-255 → 0-1)
    transforms.Normalize((0.5, 0.5, 0.5),        # normalize each RGB channel
                         (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform)
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64, shuffle=False)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples:     {len(test_dataset)}")

images, labels = next(iter(train_loader))

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([]); plt.yticks([])
    img = images[i] * 0.5 + 0.5      # undo normalization for display
    plt.imshow(img.permute(1,2,0))    # PyTorch: (C,H,W) → (H,W,C) for matplotlib
    plt.xlabel(class_names[labels[i]])
plt.show()

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        # Block 2
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.drop2 = nn.Dropout(0.25)

        # Classifier
        self.fc1   = nn.Linear(64 * 8 * 8, 512)
        self.drop3 = nn.Dropout(0.5)
        self.fc2   = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.drop2(x)

        x = x.view(-1, 64 * 8 * 8)   # flatten
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x

model = CNN()
print(model)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Training on: {device}")

# Training loop
for epoch in range(20):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear old gradients
        outputs = model(images)        # forward pass
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()                # backpropagation
        optimizer.step()               # update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch {epoch+1}/20 | Loss: {running_loss/len(train_loader):.3f} | Acc: {100*correct/total:.1f}%")
    from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

def predict_image(img_tensor):
    model.eval()
    img = img_tensor.unsqueeze(0).to(device)   # add batch dimension
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = probs.max(1)
    print(f"Prediction: {class_names[predicted.item()]} ({confidence.item()*100:.1f}% confident)")

# Test on first test image
predict_image(test_dataset[0][0])

# Save
torch.save(model.state_dict(), 'models/cifar10_model.pth')

# Load
loaded_model = CNN()
loaded_model.load_state_dict(torch.load('models/cifar10_model.pth'))
loaded_model.eval()