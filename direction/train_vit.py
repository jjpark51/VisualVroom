import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=6):
        super(VisionTransformer, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_features, num_classes)
        
        # Modify the input layer to accept grayscale images (spectrograms/MFCCs)
        self.vit.conv_proj = nn.Conv2d(1, self.vit.conv_proj.out_channels, kernel_size=16, stride=16)

    def forward(self, x):
        return self.vit(x)

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data_from_directory(data_dir, class_names, class_to_idx):
    """
    Load image paths and labels from a specific directory
    """
    image_paths = []
    labels = []
    
    # Count images per class for statistics
    class_counts = {class_name: 0 for class_name in class_names}
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    image_paths.append(img_path)
                    labels.append(class_to_idx[class_name])
                    class_counts[class_name] += 1
    
    # Print dataset statistics
    print(f"Dataset Statistics for {data_dir}:")
    total_images = sum(class_counts.values())
    for class_name, count in class_counts.items():
        if total_images > 0:
            print(f"  {class_name}: {count} images ({count/total_images*100:.1f}%)")
        else:
            print(f"  {class_name}: {count} images (0.0%)")
    print(f"Total: {total_images} images")

    return image_paths, labels

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=15, patience=5):
    """
    Train the model with early stopping
    """
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # For early stopping
    patience_counter = 0
    best_val_loss = float('inf')

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.cpu())
                scheduler.step()
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.cpu())

                # Save best model weights
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict().copy()
                    # Save the best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model_wts,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': best_acc,
                        'val_loss': epoch_loss,
                    }, 'visualvroom_best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping check
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("training_validation_loss_accuracy.png")
    plt.show()

    return model

def test_model(model, dataloader, criterion, class_names):
    """
    Evaluate the model on the test set
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    # Lists to store all predictions and true labels
    all_preds = []
    all_labels = []
    
    # Track prediction confidence
    all_confidences = []
    
    # Track samples with suspicious high confidence
    suspicious_samples = []
    
    for inputs, labels in tqdm(dataloader, desc="Testing"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Get confidence scores (probability of predicted class)
            confs = probabilities[torch.arange(probabilities.size(0)), preds]
            
            # Find suspiciously high confidence predictions
            for i, (pred, label, conf) in enumerate(zip(preds, labels, confs)):
                all_confidences.append(conf.item())
                
                # If confidence is extremely high (>0.999), flag it
                if conf.item() > 0.999:
                    suspicious_samples.append({
                        'true_class': class_names[label.item()],
                        'pred_class': class_names[pred.item()],
                        'confidence': conf.item()
                    })

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # Store predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / len(dataloader.dataset)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    # Print confidence statistics
    mean_confidence = np.mean(all_confidences)
    median_confidence = np.median(all_confidences)
    min_confidence = np.min(all_confidences)
    max_confidence = np.max(all_confidences)
    
    print(f"\nConfidence Statistics:")
    print(f"Mean confidence: {mean_confidence:.4f}")
    print(f"Median confidence: {median_confidence:.4f}")
    print(f"Min confidence: {min_confidence:.4f}")
    print(f"Max confidence: {max_confidence:.4f}")
    
    if len(suspicious_samples) > 0:
        print(f"\nWARNING: Found {len(suspicious_samples)} test samples with suspiciously high confidence (>0.999)")
        print("This might indicate memorization or data leakage.")
        print("Sample of suspicious predictions:")
        for i, sample in enumerate(suspicious_samples[:5]):
            print(f"{i+1}. True: {sample['true_class']}, Pred: {sample['pred_class']}, Conf: {sample['confidence']:.6f}")
    
    # Calculate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("confusion_matrix.png")
    plt.show()
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Calculate per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class accuracy:")
    for i, acc in enumerate(per_class_accuracy):
        print(f"{class_names[i]}: {acc:.4f}")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc.item(),
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_accuracy
    }

if __name__ == "__main__":
    # Define directories
    train_dir = "./train"
    val_dir = "./valid"
    test_dir = "./test"
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define classes
    class_names = [
        'Siren_L', 'Siren_R', 'Bike_L', 'Bike_R', 'Horn_L', 'Horn_R'
    ]
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    # Load data from each directory
    train_paths, train_labels = load_data_from_directory(train_dir, class_names, class_to_idx)
    val_paths, val_labels = load_data_from_directory(val_dir, class_names, class_to_idx)
    test_paths, test_labels = load_data_from_directory(test_dir, class_names, class_to_idx)
    
    if len(train_paths) == 0:
        raise ValueError(f"No training images found in {train_dir}")
    if len(val_paths) == 0:
        raise ValueError(f"No validation images found in {val_dir}")
    if len(test_paths) == 0:
        raise ValueError(f"No test images found in {test_dir}")
    
    # Define transformations
    # Standard transform for validation and testing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT requires 224x224 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # For grayscale images
    ])
    
    # Augmented transform for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(10),      # Random rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Create datasets
    train_dataset = ImageDataset(train_paths, train_labels, train_transform)
    val_dataset = ImageDataset(val_paths, val_labels, transform)
    test_dataset = ImageDataset(test_paths, test_labels, transform)
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    }
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the model
    model = VisionTransformer(num_classes=len(class_names)).to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-4,
        weight_decay=0.05,  # Stronger L2 regularization
        betas=(0.9, 0.999)
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=15,  # maximum number of epochs
        eta_min=1e-6  # minimum learning rate
    )
    
    # Train the model
    model = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        scheduler,
        num_epochs=15,
        patience=5  # Stop if no improvement for 5 epochs
    )
    
    # Test the model
    test_results = test_model(model, dataloaders['test'], criterion, class_names)
    
    print("\nTraining completed!")
    print(f"Final test accuracy: {test_results['test_accuracy']:.4f}")