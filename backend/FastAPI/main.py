# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from torchvision.models import vit_b_16, ViT_B_16_Weights
import librosa
from typing import List
import uvicorn
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectionalSoundViT(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vit(x)

class ModelService:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = DirectionalSoundViT(num_classes=12).to(self.device)
        self.load_model(model_path)
        self.model.eval()
        
        # Define class names
        self.class_names = [
            'Ambulance Left', 'Ambulance Middle', 'Ambulance Right',
            'Car Horn Left', 'Car Horn Middle', 'Car Horn Right',
            'Fire Truck Left', 'Fire Truck Middle', 'Fire Truck Right',
            'Police Car Left', 'Police Car Middle', 'Police Car Right'
        ]
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def process_audio(self, audio_data: List[bytes], sample_rate: int = 16000) -> Image.Image:
        """Process multiple audio channels into a single stitched image."""
        images = []
        
        for channel_data in audio_data:
            # Convert bytes to numpy array
            audio = np.frombuffer(channel_data, dtype=np.int16).astype(float) / 32768.0
            
            # Generate spectrogram
            D = librosa.stft(audio, n_fft=402, hop_length=201)
            D = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            # Generate MFCC
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=13,
                n_fft=402,
                hop_length=201
            )
            
            # Convert to images
            spec_img = self.array_to_image(D)
            mfcc_img = self.array_to_image(mfccs)
            
            # Combine images for this channel
            channel_img = self.stitch_channel_images(mfcc_img, spec_img)
            images.append(channel_img)
        
        # Stitch all channel images vertically
        return self.stitch_all_channels(images)

    def array_to_image(self, array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        # Normalize to 0-255 range
        array = ((array - array.min()) * (255.0 / (array.max() - array.min()))).astype(np.uint8)
        return Image.fromarray(array).convert('RGB')

    def stitch_channel_images(self, mfcc_img: Image.Image, spec_img: Image.Image) -> Image.Image:
        """Stitch MFCC and spectrogram images for a single channel."""
        target_width = 241
        mfcc_height = 13
        spec_height = 201
        
        # Resize images
        mfcc_img = mfcc_img.resize((target_width, mfcc_height))
        spec_img = spec_img.resize((target_width, spec_height))
        
        # Create combined image
        combined = Image.new('RGB', (target_width, mfcc_height + spec_height))
        combined.paste(mfcc_img, (0, 0))
        combined.paste(spec_img, (0, mfcc_height))
        
        return combined

    def stitch_all_channels(self, channel_images: List[Image.Image]) -> Image.Image:
        """Stitch all channel images vertically."""
        if not channel_images:
            raise ValueError("No channel images provided")
            
        channel_height = 214  # 13 + 201
        total_height = channel_height * len(channel_images)
        width = channel_images[0].width
        
        final_img = Image.new('RGB', (width, total_height))
        for i, img in enumerate(channel_images):
            final_img.paste(img, (0, i * channel_height))
        
        return final_img

    def predict(self, image: Image.Image):
        """Run inference on the processed image."""
        try:
            # Prepare image for model
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # Only return predictions with high confidence
                if confidence > 0.8:
                    predicted_name = self.class_names[predicted_class]
                    vehicle_type = predicted_name.split()[0]
                    direction = predicted_name.split()[1]
                    
                    return {
                        'vehicle_type': vehicle_type,
                        'direction': direction,
                        'confidence': float(confidence),
                        'should_notify': True
                    }
                return None
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# Initialize FastAPI app
app = FastAPI(title="Vehicle Sound Detection Model Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model service
model_service = None

@app.on_event("startup")
async def startup_event():
    global model_service
    try:
        model_service = ModelService("../model_checkpoints/best_model.pth")
        logger.info("Model service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model service: {str(e)}")
        raise

class AudioRequest(BaseModel):
    sample_rate: int = 16000

@app.post("/predict")
async def predict(
    sample_rate: int = Form(16000),
    left_channel: UploadFile = File(...),
    right_channel: UploadFile = File(...),
    rear_channel: UploadFile = File(...)
):
    try:
        # Read audio data
        audio_data = [
            await left_channel.read(),
            await right_channel.read(),
            await rear_channel.read()
        ]
        
        # Process audio into image
        processed_image = model_service.process_audio(
            audio_data, 
            sample_rate=sample_rate
        )
        
        # Get prediction
        result = model_service.predict(processed_image)
        
        if result is None:
            return {"message": "No confident prediction"}
            
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)