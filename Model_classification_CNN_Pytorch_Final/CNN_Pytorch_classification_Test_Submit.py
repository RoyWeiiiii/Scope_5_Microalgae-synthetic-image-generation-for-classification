import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
# Test Function
# -------------------
def test_model():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    TEST_DIR = r"D:\CodingProjects\machine_learning\Experiment_5_Latest\datasets_Revision_1\ratio_datasets\20%_Ori_80%_Cond_FastGAN\test"
    MODEL_PATH = r"D:\CodingProjects\machine_learning\Experiment_5_Latest\Results_Revision_1\20%_Ori_80%_Cond_FastGAN_Train_Real_Val\model.pth"
    SAVE_PATH = r"D:\CodingProjects\machine_learning\Experiment_5_Latest\Results_Revision_1\20%_Ori_80%_Cond_FastGAN_Test_Real"

    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
    ])
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    class_names = test_dataset.classes

    model = CNNClassificationModel(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_preds, all_labels = [], []
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    # Metrics
    test_accuracy = 100 * total_correct / total_samples
    test_loss = total_loss / len(test_loader)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print("\n📊 Test Metrics:")
    print(f"Accuracy  : {test_accuracy:.2f}%")
    print(f"Loss      : {test_loss:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Save metrics as CSV
    metrics_df = pd.DataFrame([{
        'Test Accuracy': test_accuracy,
        'Test Loss': test_loss,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }])
    metrics_df.to_csv(os.path.join(SAVE_PATH, "test_metrics.csv"), index=False)
    print(f"\n✅ Test metrics saved to test_metrics.csv")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Test Confusion Matrix')
    plt.savefig(os.path.join(SAVE_PATH, "test_confusion_matrix.png"))
    plt.show()

if __name__ == "__main__":
    test_model()
