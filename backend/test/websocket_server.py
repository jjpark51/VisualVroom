import asyncio
import websockets
import torch
import numpy as np
import librosa
import io
from pathlib import Path
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights
import matplotlib.pyplot as plt
from collections import deque
import json
import time
from torchvision import transforms
from threading import Lock



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

class AudioProcessor:
    def __init__(self, model_path):
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DirectionalSoundViT(num_classes=12).to(self.device)
        self.model, _, _ = load_best_model(self.model, model_path)
        self.model.eval()
        
        # Initialize buffers for each channel
        self.buffer_size = 16000 * 3  # 3 seconds of audio at 16kHz
        self.buffers = {
            0: deque(maxlen=self.buffer_size),  # Left
            1: deque(maxlen=self.buffer_size),  # Right
            2: deque(maxlen=self.buffer_size)   # Both/Rear
        }
        
        self.channel_images = {
            0: None,  # Left channel combined image
            1: None,  # Right channel combined image
            2: None   # Rear channel combined image
        }
        
        self.lock = Lock()  # For thread-safe image updates
        
        self.class_names = [
            'Ambulance Left', 'Ambulance Middle', 'Ambulance Right',
            'Car Horn Left', 'Car Horn Middle', 'Car Horn Right',
            'Fire Truck Left', 'Fire Truck Middle', 'Fire Truck Right',
            'Police Car Left', 'Police Car Middle', 'Police Car Right'
        ]

    def process_audio_chunk(self, channel, data):
        # Convert bytes to numpy array
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        
        # Add to appropriate buffer
        self.buffers[channel].extend(audio_chunk)
        
        # Check if we have enough data for processing
        if len(self.buffers[channel]) >= self.buffer_size:
            # Process this channel's buffer
            channel_img = self.process_single_channel(channel)
            
            with self.lock:
                self.channel_images[channel] = channel_img
            
            # Check if we have all three channels ready
            if all(img is not None for img in self.channel_images.values()):
                return self.process_all_channels()
        return None

    def process_single_channel(self, channel):
        # Convert buffer to numpy array
        audio_data = np.array(list(self.buffers[channel]))
        
        # Generate spectrogram
        D = librosa.stft(audio_data.astype(float)/32768.0, n_fft=402, hop_length=201)
        D = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Generate MFCC
        mfccs = librosa.feature.mfcc(
            y=audio_data.astype(float)/32768.0,
            sr=16000,
            n_mfcc=13,
            n_fft=402,
            hop_length=201
        )
        
        # Create images
        spec_img = self.create_spectrogram_image(D)
        mfcc_img = self.create_mfcc_image(mfccs)
        
        # Combine images for this channel
        return self.stitch_channel_images(mfcc_img, spec_img)

    def create_spectrogram_image(self, D):
        plt.figure(figsize=(10, 4))
        plt.imshow(D, aspect='auto', cmap='gray')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    def create_mfcc_image(self, mfccs):
        plt.figure(figsize=(10, 4))
        plt.imshow(mfccs, aspect='auto', cmap='gray')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    def stitch_channel_images(self, mfcc_img, spec_img):
        """Stitch MFCC and spectrogram images for a single channel"""
        target_width = 241
        mfcc_height = 13
        spec_height = 201
        channel_height = mfcc_height + spec_height
        
        # Resize individual images
        mfcc_img = mfcc_img.resize((target_width, mfcc_height), Image.Resampling.LANCZOS)
        spec_img = spec_img.resize((target_width, spec_height), Image.Resampling.LANCZOS)
        
        # Create combined image
        combined = Image.new('RGB', (target_width, channel_height))
        combined.paste(mfcc_img, (0, 0))
        combined.paste(spec_img, (0, mfcc_height))
        
        return combined

    def process_all_channels(self):
        """Stitch all three channel images vertically and run inference"""
        with self.lock:
            left_img = self.channel_images[0]
            rear_img = self.channel_images[2]  # Using both/rear channel
            right_img = self.channel_images[1]
            
            # Reset channel images
            self.channel_images = {0: None, 1: None, 2: None}
        
        # Create final stitched image
        target_width = 241
        channel_height = 214  # 13 + 201
        target_height = 642  # 3 * 214
        
        final_img = Image.new('RGB', (target_width, target_height))
        
        # Paste images vertically
        final_img.paste(left_img, (0, 0))
        final_img.paste(rear_img, (0, channel_height))
        final_img.paste(right_img, (0, 2 * channel_height))
        
        # Run inference on the complete stitched image
        return self.run_inference(final_img)

    def run_inference(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
        
        image = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Only send if confidence is higher than 0.8
            if confidence > 0.8:
                predicted_name = self.class_names[predicted_class]
                # Parse direction from class name
                vehicle_type = predicted_name.split()[0]  # e.g., "Ambulance"
                direction = predicted_name.split()[1]     # e.g., "Left"
                
                return {
                    'vehicle_type': vehicle_type,
                    'direction': direction,
                    'confidence': float(confidence),
                    'timestamp': time.time(),
                    'should_notify': True
                }
            return None
async def handle_websocket(websocket, path):
    processor = AudioProcessor('model_checkpoints/best_model.pth')
    
    try:
        async for message in websocket:
            # First byte indicates channel
            channel = message[0]
            audio_data = message[1:]
            
            # Process the audio chunk
            result = processor.process_audio_chunk(channel, audio_data)
            
            # Send results if available
            if result:
                await websocket.send(json.dumps(result))
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    server = await websockets.serve(
        handle_websocket,
        "0.0.0.0",
        8080,
        ping_interval=None
    )
    print("WebSocket server started")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())