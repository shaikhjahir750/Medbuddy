import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def get_data_loaders(data_dir, batch_size, img_size, val_split=0.2):
    # Define transformations for the training and validation sets
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(data_dir):
        raise ValueError(f"Dataset directory '{data_dir}' does not exist.")

    # Load dataset twice to apply different transforms using Subsets
    full_train_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    full_val_dataset = datasets.ImageFolder(data_dir, transform=val_transforms)

    num_data = len(full_train_dataset)
    val_size = int(val_split * num_data)
    train_size = num_data - val_size

    # Use a fixed generator for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset_temp = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    
    # We need the indices from the split to create proper Subsets with respective transforms
    train_indices = train_dataset.indices
    val_indices = val_dataset_temp.indices

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, full_train_dataset.classes

def build_model(num_classes):
    # Load a pre-trained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Replace the final fully connected layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    best_acc = 0.0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)

        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())

        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        # Deep copy the model if it has the best accuracy
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f'Best model saved to {save_path} with Acc: {best_acc:.4f}')

        print()

    print(f'Training complete. Best Val Acc: {best_acc:.4f}')
    return model, history

def plot_and_save_training_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    curve_path = os.path.join(save_dir, 'training_curve.png')
    plt.savefig(curve_path)
    plt.close()
    print(f"Training curves saved to {curve_path}")

def evaluate_and_save_metrics(model, val_loader, device, classes, save_dir):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Evaluating best model on validation set...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=classes)
    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=====================\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")
    print("\n" + report)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Eye Disease Detection Training")
    parser.add_argument('--data_dir', type=str, default=r'c:\Medbuddy\eye_diseases', help='Path to dataset containing class folders')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for resizing')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--save_dir', type=str, default=r'c:\Medbuddy\MODEL_TRAINING\results', help='Directory to save outputs (model, plots, reports)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'eye_disease_model.pth')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, classes = get_data_loaders(args.data_dir, args.batch_size, args.img_size, args.val_split)
    print(f"Classes found: {classes}")

    # Build model
    model = build_model(num_classes=len(classes))
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, device, save_path)
    
    # Plot Training Curves
    plot_and_save_training_curves(history, args.save_dir)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(save_path))
    evaluate_and_save_metrics(model, val_loader, device, classes, args.save_dir)

if __name__ == '__main__':
    main()
