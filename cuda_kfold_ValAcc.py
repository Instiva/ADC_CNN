import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from pathlib import Path
import random
import json
from sklearn.model_selection import KFold
from datetime import datetime

# This prototype saves the working fold files into the training_cache folder and saves the final model into the /models/ folder.
# Also, it expands the final inference eval

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Custom Dataset Class
class NEUDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Directory with 'images' folder containing class subfolders (e.g., 'crazing', 'inclusion')
        transform: PyTorch transforms for preprocessing and augmentation
        """
        self.root_dir = Path(root_dir) / 'images'
        self.transform = transform
        self.classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Images directory {self.root_dir} does not exist.")
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.is_dir():
                print(f"Warning: Directory {cls_dir} not found. Skipping.")
                continue
            for img_name in cls_dir.glob('*.jpg'):
                if img_name.is_file():
                    self.images.append(str(img_name))
                    self.labels.append(self.class_to_idx[cls])
                else:
                    print(f"Warning: {img_name} is not a valid file. Skipping.")
        if not self.images:
            raise ValueError(f"No valid images found in {self.root_dir}.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path).convert('L')  # Grayscale
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((1, 256, 256)), label

# Enhanced Data Preprocessing and Augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Evaluation Function
def evaluate_model(model, test_loader, classes):
    model.eval()
    all_preds = []
    all_labels = []
    sample_images = []
    sample_preds = []
    sample_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if len(sample_images) < 5:
                sample_images.extend(images.cpu()[:5])
                sample_preds.extend(preds.cpu()[:5])
                sample_labels.extend(labels.cpu()[:5])

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    plt.figure(figsize=(15, 5))
    for i in range(min(5, len(sample_images))):
        plt.subplot(1, 5, i+1)
        img = sample_images[i].squeeze().numpy() * 0.5 + 0.5
        plt.imshow(img, cmap='gray')
        plt.title(f'Pred: {classes[sample_preds[i]]}\nTrue: {classes[sample_labels[i]]}')
        plt.axis('off')
    plt.show()

    # Output confusion matrix in requested format
    print("Confusion Matrix:")
    print("[")
    for row in cm:
        print(" [" + ", ".join(map(str, row)) + "],")
    print("]")

# Data Loading  (Relative to script's directory)
script_dir = Path(__file__).parent
data_root = script_dir / 'kaggle' / 'NEU-DET'
train_dataset = NEUDataset(root_dir=data_root / 'train', transform=train_transform)
val_dataset = NEUDataset(root_dir=data_root / 'validation', transform=val_test_transform)
full_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])  # Combine for k-fold

# Compute Class Weights
class_counts = [len(list((data_root / 'train' / 'images' / cls).glob('*.jpg'))) + 
                len(list((data_root / 'validation' / 'images' / cls).glob('*.jpg'))) 
                for cls in train_dataset.classes]
class_weights = torch.tensor([1.0 / np.sqrt(count) if count > 0 else 0 for count in class_counts], dtype=torch.float)

# CNN Model Definition
class DefectCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(DefectCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 128x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 64x64
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # 32x32
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Deeper layer
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2)   # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# k-fold Cross-Validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
best_models = []
fold_results = []

# Define paths for saving models
models_dir = script_dir / 'models'
os.makedirs(models_dir, exist_ok=True)  # Ensure models directory exists
training_cache_dir = models_dir / 'training_cache'
os.makedirs(training_cache_dir, exist_ok=True)  # Ensure training_cache folder exists
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
final_save_path = os.path.join(models_dir, f'defect_classifier_(val_acc)_{timestamp}.pth')  # Define final save path for the rolled-up model. Will be used after k-fold and eval

for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
    print(f'\nFold {fold + 1}/{k_folds}')
    train_subsampler = Subset(full_dataset, train_ids)
    val_subsampler = Subset(full_dataset, val_ids)
    train_loader = DataLoader(train_subsampler, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=16, shuffle=False)

    model = DefectCNN(num_classes=6).to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.85, patience=5) # This will reduce the learning rate by 15% after {patience} consecutive epochs without validation loss improvement
    scaler = GradScaler('cuda')

    # Training Loop with Early Stopping and TensorBoard
    writer = SummaryWriter(f'runs/fold_{fold + 1}')
    best_val_acc = 0.0 # Initializes at zero so acc can increase
    train_patience = 15
    patience_counter = 0
    for epoch in range(100):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast('cuda'):
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', lr, epoch)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        metrics = {"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc, "lr": lr}
        metrics_path = os.path.join(training_cache_dir, f"training_metrics_fold_{fold +1}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        print(f'Epoch {epoch+1}/100, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {lr:.6f}')
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_path = os.path.join(training_cache_dir, f'best_model_fold_{fold + 1}_(val_acc).pth')
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= train_patience:
                print("Early stopping triggered")
                break
    writer.close()
    best_models.append(best_model_path)  # Use the actual path
    fold_results.append({'fold': fold + 1, 'val_loss': val_loss, 'val_acc': best_val_acc})

# Evaluation on Test Set with Best Model
test_dataset = NEUDataset(root_dir=data_root / 'validation', transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
best_model_path = max(best_models, key=lambda p: fold_results[best_models.index(p)]['val_acc']) # Use max val_acc for consistency with early stopping behavior.
model.load_state_dict(torch.load(best_model_path, weights_only=True))
model.eval()
evaluate_model(model, test_loader, train_dataset.classes)

# Save Final Model
torch.save(model.state_dict(), final_save_path)
print(f"Model saved successfully at: {final_save_path}")
print("Performing inference on sample images...")
model.eval()
with torch.no_grad():
    num_batches_to_show = 15 # Show batches for variety
    for i, (images, labels) in enumerate(test_loader):
        if i >= num_batches_to_show:
            break
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for j in range(images.size(0)): # Process all images in the batch
            img = images[j].cpu().squeeze().numpy() * 0.5 + 0.5
            true_class = train_dataset.classes[labels[j]]
            pred_class = train_dataset.classes[preds[j]]
            print(f"Batch {i+1}, Sample {j+1}: Predicted: {pred_class}, True: {true_class}")
            if j >= 5: # Limit to 6 samples per batch
                break