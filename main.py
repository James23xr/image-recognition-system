import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import time
import multiprocessing
from tqdm import tqdm

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform_train, transform_test

def load_data(transform_train, transform_test):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

class ImprovedResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedResNet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

def train(model, trainloader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        for data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(testloader, desc='Evaluating')
        for data in pbar:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'accuracy': f'{100 * correct / total:.2f}%'})
    return 100 * correct / total

def main():
    device = get_device()
    transform_train, transform_test = get_transforms()
    trainloader, testloader = load_data(transform_train, transform_test)

    model = ImprovedResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting initial evaluation...")
    start_time = time.time()
    initial_accuracy = evaluate(model, testloader, device)
    initial_time = time.time() - start_time
    print(f"Initial accuracy: {initial_accuracy:.2f}%, Time taken: {initial_time:.2f} seconds")

    print("Starting training...")
    train(model, trainloader, criterion, optimizer, device, epochs=10)
    print("Training complete.")

    print("Starting final evaluation...")
    start_time = time.time()
    improved_accuracy = evaluate(model, testloader, device)
    improved_time = time.time() - start_time
    print(f"Final accuracy: {improved_accuracy:.2f}%, Time taken: {improved_time:.2f} seconds")

    accuracy_improvement = improved_accuracy - initial_accuracy
    time_improvement = (initial_time - improved_time) / initial_time * 100

    print(f"Accuracy improved by {accuracy_improvement:.2f}%")
    print(f"Processing time reduced by {time_improvement:.2f}%")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()