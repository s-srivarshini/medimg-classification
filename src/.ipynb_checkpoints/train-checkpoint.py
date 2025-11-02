import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data import get_dataloaders
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ✅ Auto select GPU / MPS / CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using NVIDIA GPU")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (GPU not detected)")

def train_model(data_dir, num_epochs=5, lr=0.0001):
    train_loader, val_loader, _ = get_dataloaders(data_dir)

    # ✅ Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        print(f"✅ Train Accuracy: {train_acc:.4f} | Loss: {running_loss/len(train_loader):.4f}")

    # ✅ Save model
    torch.save(model.state_dict(), "../checkpoints/model.pth")
    print("🎯 Model saved to ../checkpoints/model.pth")

if __name__ == "__main__":
    train_model("../data/chest_xray")
