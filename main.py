import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit

# Definición del modelo
class AffectNetCNN(nn.Module):
    def __init__(self, num_classes=2, lr=0.001):
        super(AffectNetCNN, self).__init__()
        self.conv1_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2_layer = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.conv1_layer(x)
        x = self.conv2_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        x = F.softmax(x, dim=1)
        return x
    
    # Definición de la función de perdida
    def _loss_(self, x, target):
        output = self.forward(x)
        loss = F.cross_entropy(output, target)
        return loss
    
    # Definición de la función de entrenamiento
    def train(self, data_loader, epochs = 10):
        for epoch in range(epochs):
            for x, target in data_loader:
                self.optimizer.zero_grad()
                loss = self._loss_(x, target)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    
    # Definición de la función de evaluación
    def test(self, data_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, target in data_loader:
                output = self.forward(x)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return correct / total

# Definición del transformador  
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cargar el conjunto de datos
dataset = datasets.ImageFolder('AffectNet', transform=transform)
num_classes = len(dataset.classes)
print('Dataset classes:', dataset.classes)

# Obtener las etiquetas para la estratificación
targets = [label for _, label in dataset]

# Crear índices de train y test de manera estratificada
test_size = 0.2 # Porcentaje de datos para el conjunto de prueba
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
train_idx, test_idx = next(stratified_split.split(np.arange(len(targets)), targets))

# Crear samplers para los DataLoaders
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

# Crear los DataLoaders con los samplers
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)

# Inicializar y entrenar el modelo
model = AffectNetCNN(num_classes=num_classes)

print('Training CNN model...')
model.train_model(train_loader, epochs=10)

# Evaluación del modelo en el conjunto de prueba
accuracy = model.test(test_loader)
print(f'Test Accuracy: {accuracy:.2f}')

# Guardar el modelo entrenado
torch.save(model.state_dict(), 'model.pth')