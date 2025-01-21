import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
class DirectionalSoundViT(nn.Module):
    def __init__(self, num_classes=12):  # Updated to 12 classes
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vit(x)
    
def load_best_model(model, filepath):
    """Load the best model weights"""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"No model checkpoint found at {filepath}")
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['val_acc']:.2f}%")
    return model, checkpoint['epoch'], checkpoint['val_acc']


def inference(model, image_path, device=None):
    """Run inference on a single image"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    # Define class names
    class_names = [
        'Ambulance Left', 'Ambulance Middle', 'Ambulance Right',
        'Car Horn Left', 'Car Horn Middle', 'Car Horn Right',
        'Fire Truck Left', 'Fire Truck Middle', 'Fire Truck Right',
        'Police Car Left', 'Police Car Middle', 'Police Car Right'
    ]
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
    return {
        'predicted_class': class_names[predicted_class],
        'confidence': confidence * 100,
        'all_probabilities': {
            class_name: prob.item() * 100 
            for class_name, prob in zip(class_names, probabilities[0])
        }
    }


def test_model_inference():
    # Load the model
    model = DirectionalSoundViT(num_classes=12)
    print("\nLoading best model for evaluation...")
    model, best_epoch, best_val_acc = load_best_model(model, '../model_checkpoints/best_model.pth')
    print(f"Loaded model from epoch {best_epoch} with validation accuracy {best_val_acc:.2f}%")

    # Run example inference
    print("\nRunning example inference...")
    test_image_path = "./single_output/sound_1_channel.png"
    # test_image_path = "./test/test_output/final_stitched.png"

    
    if Path(test_image_path).exists():
        # Run multiple inference iterations
        num_runs = 10
        times = []
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            result = inference(model, test_image_path)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        import statistics
        avg_time = statistics.mean(times) * 1000
        std_dev = statistics.stdev(times) * 1000


        print(f"\nInference results:")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"\nInference Time:")
        print(f"Average: {avg_time:.4f} milliseconds")
        print(f"Standard Deviation: {std_dev:.4f} milliseconds")
        print("\nAll class probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"{class_name}: {prob:.2f}%")
    else:
        print(f"Error: Test image not found at {test_image_path}")
        


if __name__ == "__main__":
    test_model_inference()