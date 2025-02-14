import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from vit_pytorch import SimpleViT
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MetricTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def update(self, train_loss, val_loss, train_acc, val_acc):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
    
    def plot_metrics(self, save_dir='plots'):
        Path(save_dir).mkdir(exist_ok=True)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        epochs = range(1, len(self.train_losses) + 1)
        
        # Plot losses
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_metrics.png')
        plt.close()

class DirectionalSoundDataset(Dataset):
    def __init__(self, base_dir, transform=None, target_size=(240, 640)):  # Adjusted size
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.target_size = target_size
        
        self.class_to_idx = {
            'ambulance_L': 0, 'ambulance_R': 1,
            'carhorns_L': 2, 'carhorns_R': 3,
            'FireTruck_L': 4, 'FireTruck_R': 5,
            'policecar_L': 6, 'policecar_R': 7
        }
        
        self.files = []
        self.labels = []
        
        for class_name in self.class_to_idx.keys():
            class_dir = self.base_dir / class_name
            if class_dir.exists():
                class_files = list(class_dir.glob(f"{class_name}_*.png"))
                self.files.extend(class_files)
                self.labels.extend([self.class_to_idx[class_name]] * len(class_files))
        
        if len(self.files) == 0:
            raise RuntimeError(f"No spectrogram files found in {base_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]
        
        spectrogram = Image.open(img_path).convert('RGB')
        if spectrogram.size != self.target_size:
            spectrogram = spectrogram.resize(self.target_size)
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram, label

def create_data_loaders(base_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    
    full_dataset = DirectionalSoundDataset(
        base_dir=base_dir,
        transform=transform,
        target_size=(240, 640)
    )
    
    # Calculate split sizes for 70-10-20 split
    total_size = len(full_dataset)
    test_size = int(0.2 * total_size)  # 20%
    val_size = int(0.1 * total_size)   # 10%
    train_size = total_size - test_size - val_size  # 70%
    
    # Create indices for splits
    indices = list(range(total_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"\nDataset split sizes:")
    print(f"Total dataset size: {total_size}")
    print(f"Training set size: {train_size} ({train_size/total_size*100:.1f}%)")
    print(f"Validation set size: {val_size} ({val_size/total_size*100:.1f}%)")
    print(f"Test set size: {test_size} ({test_size/total_size*100:.1f}%)\n")
    
    return train_loader, val_loader, test_loader

# Initialize model with reduced dimensions
model = SimpleViT(
    image_size = (240, 640),
    patch_size = 16,
    num_classes = 8,
    dim = 512,        # Reduced from 1024
    depth = 6,
    heads = 8,        # Reduced from 16
    mlp_dim = 1024    # Reduced from 2048
)

# Training functions
def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights after accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar.set_postfix({
            'Loss': total_loss/(batch_idx+1),
            'Acc': 100.*correct/total
        })
    
    return total_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device, compute_confusion=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Lists to store all predictions and targets if computing confusion matrix
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if compute_confusion:
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
    
    if compute_confusion:
        return total_loss/len(val_loader), 100.*correct/total, all_predictions, all_targets
    return total_loss/len(val_loader), 100.*correct/total

def compute_per_class_metrics(predictions, targets, class_names):
    """Compute per-class accuracy and create confusion matrix visualization."""
    # Convert lists to numpy arrays if they aren't already
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(targets, predictions)
    
    # Compute per-class accuracy
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1) * 100
    
    # Create confusion matrix plot with better formatting
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print per-class accuracy with better formatting
    print("\nPer-class Test Accuracy:")
    for class_name, accuracy in zip(class_names, per_class_accuracy):
        print(f"{class_name:12s}: {accuracy:6.2f}%")
    
    return per_class_accuracy

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    base_dir="Dataset of warning sound types and source directions",
    batch_size=8  # Reduced from 32
)

# Initialize metric tracker
metric_tracker = MetricTracker()

# Training loop
num_epochs = 30
best_val_acc = 0
best_model = None

print(f"Training on {device}")
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()
    
    # Update metrics
    metric_tracker.update(train_loss, val_loss, train_acc, val_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model.state_dict()
        torch.save(best_model, 'best_vit_model.pth')
        print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")

# Plot training metrics
metric_tracker.plot_metrics()

# Final test evaluation
print("\nEvaluating final model on test set...")
model.load_state_dict(torch.load('best_vit_model.pth'))
test_loss, test_acc, test_predictions, test_targets = validate(model, test_loader, criterion, device, compute_confusion=True)

# Define class names in order
class_names = [
    'ambulance_L', 'ambulance_R',
    'carhorns_L', 'carhorns_R',
    'FireTruck_L', 'FireTruck_R',
    'policecar_L', 'policecar_R'
]

# Compute and display per-class metrics
per_class_acc = compute_per_class_metrics(test_predictions, test_targets, class_names)
print(f"\nOverall Test Accuracy: {test_acc:.2f}%")