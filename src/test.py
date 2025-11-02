import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(data_path):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(os.path.join(data_path, "test"), transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("../checkpoints/model.pth"))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f"✅ Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    test_model("../data/chest_xray")
