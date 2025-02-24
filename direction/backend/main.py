from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import librosa
import io
import soundfile as sf
import torch.nn as nn
from torchvision.models import vit_b_16
import logging
import json
import asyncio
from typing import List, Dict
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
SAMPLE_RATE = 16000
N_FFT = 402
HOP_LENGTH = 201
N_MFCC = 13

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=6):
        super(VisionTransformer, self).__init__()
        self.vit = vit_b_16()
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_features, num_classes)
        
        # Modify the input layer to accept grayscale images
        self.vit.conv_proj = nn.Conv2d(1, self.vit.conv_proj.out_channels, 
                                      kernel_size=16, stride=16)

    def forward(self, x):
        return self.vit(x)

class AudioProcessor:
    def __init__(self, model_path='./checkpoints/best_model_checkpoint.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def process_audio_channels(self, left_channel, right_channel, sample_rate=16000):
        """Process audio channels exactly like inference.py"""
        try:
            # Convert bytes to float32 arrays (-1 to 1 range)
            left_audio = self.bytes_to_audio(left_channel, sample_rate)
            right_audio = self.bytes_to_audio(right_channel, sample_rate)

            # Generate spectrograms with exact same parameters as inference.py
            left_spectrogram = self.convert_to_spectrogram(
                left_audio, 
                sr=sample_rate,
                n_fft=402,  # Match inference.py
                hop_length=201  # Match inference.py
            )
            right_spectrogram = self.convert_to_spectrogram(
                right_audio, 
                sr=sample_rate,
                n_fft=402,
                hop_length=201
            )

            # Generate MFCCs with exact same parameters
            left_mfcc = self.convert_to_mfcc(
                left_audio,
                sr=sample_rate,
                n_mfcc=13,  # Match inference.py
                n_fft=402,
                hop_length=201,
                fmax=sample_rate/2  # Match inference.py Nyquist frequency
            )
            right_mfcc = self.convert_to_mfcc(
                right_audio,
                sr=sample_rate,
                n_mfcc=13,
                n_fft=402,
                hop_length=201,
                fmax=sample_rate/2
            )

            # Convert to images with exact dimensions
            left_spectrogram_img = self.array_to_image(left_spectrogram, 241, 201)
            right_spectrogram_img = self.array_to_image(right_spectrogram, 241, 201)
            left_mfcc_img = self.array_to_image(left_mfcc, 241, 13)
            right_mfcc_img = self.array_to_image(right_mfcc, 241, 13)

            # Create final image exactly like inference.py
            final_img = Image.new('L', (241, 428))
            final_img.paste(left_mfcc_img, (0, 0))
            final_img.paste(left_spectrogram_img, (0, 13))
            final_img.paste(right_mfcc_img, (0, 214))
            final_img.paste(right_spectrogram_img, (0, 227))

            return final_img

        except Exception as e:
            logger.error(f"Error processing audio channels: {e}")
            raise

    def array_to_image(self, array, width, height):
        """Convert array to image exactly like inference.py"""
        # Ensure array is finite
        array = np.nan_to_num(array)
        
        # Normalize exactly like inference.py
        array = ((array - array.min()) * (255.0 / (array.max() - array.min() + 1e-8))).astype(np.uint8)
        image = Image.fromarray(array).convert('L')
        image = image.resize((width, height))
        return image

    def convert_to_spectrogram(self, audio, sr, n_fft=402, hop_length=201):
        """Convert to spectrogram exactly like inference.py"""
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        return D_db

    def convert_to_mfcc(self, audio, sr, n_mfcc=13, n_fft=402, hop_length=201, fmax=None):
        """Convert to MFCC exactly like inference.py"""
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=n_mfcc, 
            n_fft=n_fft, 
            hop_length=hop_length,
            fmax=fmax
        )
        return mfccs
    def _load_model(self, model_path):
        """Load model exactly like inference.py"""
        try:
            # Initialize the model
            model = VisionTransformer(num_classes=6)
            
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model state dict from checkpoint
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['best_acc']:.4f}")
            else:
                # Fallback for older model format
                model.load_state_dict(checkpoint)
                logger.info("Loaded legacy model format")
            
            # Move model to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def bytes_to_audio(self, audio_bytes, sample_rate=16000):
        """Convert raw PCM bytes to numpy array."""
        try:
            # Convert bytes to numpy array (16-bit integers)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Normalize to float between -1 and 1
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            return audio_array
        except Exception as e:
            logger.error(f"Error converting bytes to audio: {e}")
            raise
            
    def predict(self, image):
        """Make prediction using the model."""
        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_class = torch.max(probabilities, 1)

            # Map class index to vehicle type and direction
            classes = ['Siren_L', 'Siren_R', 'Bike_L', 'Bike_R', 'Horn_L', 'Horn_R']
            predicted_class = classes[top_class.item()]
            vehicle_type, direction = predicted_class.split('_')
            
            confidence = float(top_prob.item())
            prediction = {
                'vehicle_type': vehicle_type,
                'direction': 'Left' if direction == 'L' else 'Right',
                'confidence': confidence,
                'should_notify': confidence > 0.97
            }
            
            # Only log predictions with confidence above 0.9
            if confidence > 0.97:
                logger.info(f"High confidence prediction: {prediction}")
            
            return prediction

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
# Initialize audio processor
audio_processor = AudioProcessor()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive audio data as JSON with base64 encoded audio
            data = await websocket.receive_json()
            
            try:
                # Convert base64 audio data to numpy arrays
                left_channel = audio_processor.base64_to_audio(data['left_channel'])
                right_channel = audio_processor.base64_to_audio(data['right_channel'])

                # Process audio and get feature image
                feature_image = audio_processor.process_audio_channels(
                    left_channel, right_channel)

                # Make prediction
                prediction = audio_processor.predict(feature_image)
                
                # Only send prediction if confidence is high enough
                if prediction['confidence'] > 0.9:
                    await websocket.send_json(prediction)
                else:
                    await websocket.send_json({
                        "message": "No confident prediction available"
                    })

            except Exception as e:
                logger.error(f"Error processing audio data: {e}")
                await websocket.send_json({
                    'error': str(e)
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Also keep a REST endpoint for compatibility
@app.post("/predict")
async def predict(
    left_channel: UploadFile = File(...),
    right_channel: UploadFile = File(...),
    sample_rate: int = Form(16000)
):
    try:
        # Read raw audio data from files
        left_data = await left_channel.read()
        right_data = await right_channel.read()

        logger.info(f"Received audio data - Left: {len(left_data)} bytes, Right: {len(right_data)} bytes")

        # Process audio and get feature image
        feature_image = audio_processor.process_audio_channels(
            left_data, right_data, sample_rate)

        # Make prediction
        prediction = audio_processor.predict(feature_image)
        
        # Check confidence threshold
        if prediction['confidence'] > 0.9:
            logger.info(f"High confidence prediction: {prediction}")
            return prediction
        else:
            logger.info(f"Low confidence prediction, returning message: {prediction['confidence']}")
            return {"message": "No confident prediction available"}

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return {"error": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)