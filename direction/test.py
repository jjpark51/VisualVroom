import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import vit_b_16
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Define the model architecture (must match the one used for training)
class VisionTransformer(nn.Module):
    def __init__(self, num_classes=6):
        super(VisionTransformer, self).__init__()
        self.vit = vit_b_16(weights=None)  # No pretrained weights needed when loading checkpoint
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_features, num_classes)
        
        # Modify the input layer to accept grayscale images
        self.vit.conv_proj = nn.Conv2d(1, self.vit.conv_proj.out_channels, kernel_size=16, stride=16)

    def forward(self, x):
        return self.vit(x)

# Define dataset class for test images
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

# Function to load data from a directory
def load_test_data(test_dir):
    image_paths = []
    labels = []
    class_names = [
        'Siren_L', 'Siren_R', 'Bike_L', 'Bike_R', 'Horn_L', 'Horn_R'
    ]
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    image_paths.append(img_path)
                    labels.append(class_to_idx[class_name])

    return image_paths, labels, class_names

# Function to test the model
def test_model(model, dataloader, criterion, class_names):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    # Lists to store all predictions and true labels
    all_preds = []
    all_labels = []
    
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # Store predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / len(dataloader.dataset)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    # Calculate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("test_confusion_matrix.png")
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

# Function to visualize incorrect predictions
def visualize_incorrect_predictions(model, dataloader, class_names, num_images=10):
    model.eval()
    incorrect_samples = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        # Find incorrect predictions
        incorrect_mask = preds != labels.data
        incorrect_indices = torch.nonzero(incorrect_mask, as_tuple=True)[0]
        
        for idx in incorrect_indices:
            incorrect_samples.append({
                'image': inputs[idx].cpu(),
                'true_label': labels[idx].item(),
                'pred_label': preds[idx].item()
            })
            
            if len(incorrect_samples) >= num_images:
                break
        
        if len(incorrect_samples) >= num_images:
            break
    
    # Plot incorrect predictions
    if incorrect_samples:
        fig, axes = plt.subplots(2, 5, figsize=(15, 8)) if num_images >= 10 else plt.subplots(1, num_images, figsize=(15, 4))
        axes = axes.flatten() if num_images >= 10 else axes
        
        for i, sample in enumerate(incorrect_samples[:num_images]):
            img = sample['image'][0].numpy()  # Get the first channel (grayscale)
            true_label = class_names[sample['true_label']]
            pred_label = class_names[sample['pred_label']]
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("incorrect_predictions.png")
        plt.show()
    else:
        print("No incorrect predictions found in the samples.")

# Main execution
if __name__ == "__main__":
    # Set the path to your test data directory
    test_dir = "./test
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the test data
    image_paths, labels, class_names = load_test_data(test_dir)
    
    # Print information about the test data
    print(f"Number of test images: {len(image_paths)}")
    print(f"Classes: {class_names}")
    
    # Define data transformation for test images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Normalization for grayscale
    ])
    
    # Create test dataset and dataloader
    test_dataset = ImageDataset(image_paths, labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize the model
    model = VisionTransformer(num_classes=len(class_names)).to(device)
    
    # Load the saved checkpoint
    checkpoint_path = "./backend/checkpoints/feb_25_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint at epoch {checkpoint['epoch']} with validation accuracy {checkpoint['best_acc']:.4f}")
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Test the model
    test_results = test_model(model, test_loader, criterion, class_names)
    
    # Visualize some incorrect predictions
    visualize_incorrect_predictions(model, test_loader, class_names)
    
    # Save test results
    results_summary = {
        'test_accuracy': test_results['test_accuracy'],
        'per_class_accuracy': {class_names[i]: acc for i, acc in enumerate(test_results['per_class_accuracy'])}
    }
    
    print("\nTest Results Summary:")
    print(f"Overall Accuracy: {results_summary['test_accuracy']:.4f}")
    print("Per-class Accuracy:")
    for class_name, acc in results_summary['per_class_accuracy'].items():
        print(f"  {class_name}: {acc:.4f}")