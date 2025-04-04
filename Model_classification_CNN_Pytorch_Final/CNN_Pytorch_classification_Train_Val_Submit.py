#################### Classification using CNN with Pytorch - For Training & Validation ####################
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

# -------------------
# CNN Model
# -------------------
class CNNClassificationModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNClassificationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(6400, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# -------------------
# Training Function
# -------------------
def train_classification():
    NUM_EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DATASET_DIR = r"D:\CodingProjects\machine_learning\Experiment_5_Latest\datasets_Revision_1\ratio_datasets\20%_Ori_80%_Cond_FastGAN"
    SAVE_PATH = r"D:\CodingProjects\machine_learning\Experiment_5_Latest\Results_Revision_1\20%_Ori_80%_Cond_FastGAN_Train_Real_Val"
    os.makedirs(SAVE_PATH, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_DIR, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(DATASET_DIR, "val"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loaded {len(train_dataset)} training images.")
    print(f"Loaded {len(val_dataset)} validation images.")

    # Show actual image size after transform
    sample_img, _ = train_dataset[0]
    print(f"Image tensor shape after transform (C x H x W): {sample_img.shape}")

    model = CNNClassificationModel(num_classes=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)   #Include learning rate scheduler#
    criterion = nn.CrossEntropyLoss()

    accuracy_metric = Accuracy(task="multiclass", num_classes=3).to(DEVICE)
    precision_metric = Precision(task="multiclass", num_classes=3, average='weighted').to(DEVICE)
    recall_metric = Recall(task="multiclass", num_classes=3, average='weighted').to(DEVICE)
    f1_metric = F1Score(task="multiclass", num_classes=3, average='weighted').to(DEVICE)

    metrics_log = []
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []
    learning_rates = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_preds, train_true = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(labels.cpu().numpy())

        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds, val_true = [], []

        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

                accuracy_metric.update(predicted, labels)
                precision_metric.update(predicted, labels)
                recall_metric.update(predicted, labels)
                f1_metric.update(predicted, labels)

        val_accuracy = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        f1 = f1_metric.compute().item()

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        print(f"Epoch {epoch+1}: Train Acc = {train_accuracy:.4f}%, Val Acc = {val_accuracy:.4f}%, LR = {current_lr:.6f}")

        metrics_log.append([epoch+1, train_accuracy, train_loss, val_accuracy, val_loss, precision, recall, f1, current_lr])

        scheduler.step()

    torch.save(model.state_dict(), os.path.join(SAVE_PATH, "model.pth"))
    print(f"Model weights saved to {os.path.join(SAVE_PATH, 'model.pth')}")

    df = pd.DataFrame(metrics_log, columns=["Epoch", "Train Accuracy", "Train Loss", "Val Accuracy", "Val Loss", "Precision", "Recall", "F1", "Learning Rate"])
    df.to_csv(os.path.join(SAVE_PATH, "metrics.csv"), index=False)

    # Visualisation plots

    # Training Vs Validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "loss_curve.png"))
    plt.close()

    # Training Vs Validation accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "accuracy_curve.png"))
    plt.close()

    # Confusion matrices
    class_names = train_dataset.classes
    train_cm = confusion_matrix(train_true, train_preds, labels=list(range(len(class_names))))
    val_cm = confusion_matrix(val_true, val_preds, labels=list(range(len(class_names))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Train Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(SAVE_PATH, "train_confusion_matrix.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(val_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(SAVE_PATH, "val_confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    train_classification()
