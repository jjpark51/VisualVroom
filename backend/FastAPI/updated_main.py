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
    def __init__(self, num_classes=15):  # Updated to 15 classes
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_features, num_classes)
        
        # Enable gradient checkpointing for memory efficiency
        self.vit.encoder.gradient_checkpointing = True

    def forward(self, x):
        return self.vit(x)

class ModelService:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model with 15 classes
        self.model = DirectionalSoundViT(num_classes=15).to(self.device)
        self.load_model(model_path)
        self.model.eval()
        
        # Updated class names to include bike categories
        self.class_names = [
            'Ambulance Left', 'Ambulance Middle', 'Ambulance Right',
            'Car_Horn Left', 'Car_Horn Middle', 'Car_Horn Right',
            'Fire_Truck Left', 'Fire_Truck Middle', 'Fire_Truck Right',
            'Police_Car Left', 'Police_Car Middle', 'Police_Car Right',
            'Bike Left', 'Bike Right', 'Bike Back'  # Added bike categories
        ]
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

         # Add thresholds for signal validation
        self.energy_threshold = 0.001  # Minimum energy threshold
        self.noise_threshold_db = 0  # Noise floor in dB
        self.min_confidence = 0.8  # Minimum confidence for prediction
        self.high_confidence = 0.9  # High confidence threshold
        self.prediction_history = []  # Store recent predictions
        self.history_size = 3  # Number of predictions to consider for temporal consistency
    
    def check_signal_quality(self, audio: np.ndarray) -> bool:
        """
        Check if the audio signal meets quality criteria with adjusted thresholds.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            bool: True if signal meets quality criteria, False otherwise
        """
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio**2))
        
        # Calculate signal statistics
        peak = np.max(np.abs(audio))
        crest_factor = peak / (np.sqrt(np.mean(audio**2)) + 1e-10)
        
        # Log detailed signal information
    #    logger.info(f"Signal metrics - Energy: {energy:.6f}, Peak: {peak:.6f}, Crest factor: {crest_factor:.2f}")
        
        # Adjusted thresholds
        energy_threshold = 0.001  # Lowered energy threshold
        if energy < energy_threshold:
    #        logger.info(f"Low signal energy: {energy:.6f}")
            return False
            
        # Adjusted crest factor threshold
        if crest_factor > 30:  # Increased threshold for crest factor
     #       logger.info(f"High crest factor (potential noise): {crest_factor:.2f}")
            return False
        
        # Add zero-crossing rate check
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)
    #    logger.info(f"Zero-crossing rate: {zero_crossings:.4f}")
        
        if zero_crossings > 0.9:  # Too many zero crossings might indicate noise
     #       logger.info(f"High zero-crossing rate: {zero_crossings:.4f}")
            return False
        
        return True
    
    def analyze_spectral_content(self, D: np.ndarray) -> bool:
        """
        Analyze spectral content of the signal with adjusted thresholds.
        
        Args:
            D: STFT magnitude spectrum
            
        Returns:
            bool: True if spectral content is valid, False otherwise
        """
        # Convert to dB scale
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Calculate statistics
        mean_magnitude = np.mean(D_db)
        std_magnitude = np.std(D_db)
        max_magnitude = np.max(D_db)
        
        # Log detailed signal statistics
       # logger.info(f"Signal statistics - Mean: {mean_magnitude:.2f} dB, Std: {std_magnitude:.2f}, Max: {max_magnitude:.2f} dB")
        
        # Adjusted thresholds
        if mean_magnitude < -70:  # Much lower threshold for mean magnitude
        #    logger.info(f"Very weak signal: {mean_magnitude:.2f} dB")
            return False
            
        if std_magnitude < 3:  # Lower variation threshold
         #   logger.info(f"Very low spectral variation: {std_magnitude:.2f}")
            return False
            
        # Check if there are any strong frequency components
        peak_threshold = -40  # Adjusted peak threshold
        if max_magnitude < peak_threshold:
        #    logger.info(f"No strong frequency components detected: max {max_magnitude:.2f} dB")
            return False
        
        return True
        

    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle both state_dict and full checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info("Model loaded successfully")
            
            # Clear CUDA cache after loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def process_audio(self, audio_data: List[bytes], sample_rate: int = 16000) -> tuple[Image.Image, bool]:
        """Process audio and check signal quality with more lenient validation."""
        images = []
        channels_valid = []
        
        for i, channel_data in enumerate(audio_data):
            try:
        #        logger.info(f"\nProcessing channel {i+1}:")
                # Convert bytes to numpy array
                audio = np.frombuffer(channel_data, dtype=np.int16).astype(float) / 32768.0
                
                # Check signal quality
                signal_valid = self.check_signal_quality(audio)
                channels_valid.append(signal_valid)
                
                # Generate spectrogram
                D = librosa.stft(audio, n_fft=402, hop_length=201)
                
                # Check spectral content
                spectral_valid = self.analyze_spectral_content(D)
                channels_valid[-1] &= spectral_valid
                
                # Continue processing if either check passed
                if signal_valid or spectral_valid:
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
                
            except Exception as e:
        #        logger.error(f"Error processing channel {i+1}: {str(e)}")
                channels_valid.append(False)
                continue
        
        # Consider signal valid if at least one channel has good signal
        is_valid_signal = any(channels_valid)
        
        if not is_valid_signal:
     #       logger.info("All channels failed quality checks")
            return None, False
        
        if len(images) == 0:
     #       logger.info("No valid images generated from channels")
            return None, False
            
        # Log final decision
    #    logger.info(f"Signal validated: {is_valid_signal}, Valid channels: {sum(channels_valid)}/{len(channels_valid)}")
        
        # Stitch all channel images vertically
        return self.stitch_all_channels(images), True
    
    def check_temporal_consistency(self, prediction: dict) -> bool:
        """
        Check if prediction is consistent with recent history.
        
        Args:
            prediction: Current prediction dictionary
            
        Returns:
            bool: True if prediction is consistent, False otherwise
        """
        if not self.prediction_history:
            self.prediction_history.append(prediction)
            return True
            
        # Check if prediction matches recent history
        recent_predictions = self.prediction_history[-self.history_size:]
        matching_predictions = sum(
            1 for p in recent_predictions
            if p['vehicle_type'] == prediction['vehicle_type'] and 
            p['direction'] == prediction['direction']
        )
        
        # Update history
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
            
        # Require majority agreement
        return matching_predictions >= (len(recent_predictions) // 2)
    
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
        """Run inference with reliability checks."""
        try:
            if image is None:
                return None
                
            # Clear CUDA cache before prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Prepare image for model
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get top k predictions
                k = 2
                top_probs, top_indices = torch.topk(probabilities, k)
                
                # Check confidence margins
                confidence_margin = top_probs[0][0] - top_probs[0][1]
                
                predicted_class = top_indices[0][0].item()
                confidence = top_probs[0][0].item()
                
                # Apply stricter confidence thresholds
                if confidence > self.min_confidence:
                    predicted_name = self.class_names[predicted_class]
                    vehicle_type = predicted_name.split()[0]
                    direction = ' '.join(predicted_name.split()[1:])
                    
                    prediction = {
                        'vehicle_type': vehicle_type,
                        'direction': direction,
                        'confidence': float(confidence),
                        'should_notify': confidence > self.high_confidence
                    }
                    
                    # Check temporal consistency
                    if confidence > self.high_confidence or self.check_temporal_consistency(prediction):
                        if confidence > 0.98:
                            print(prediction)
                        return prediction
                    else:
                        logger.info("Prediction failed temporal consistency check")
                        
                return None
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        model_service = ModelService("../model_checkpoints/")
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
        
        # Process audio into image with signal quality check
        processed_image, is_valid = model_service.process_audio(
            audio_data, 
            sample_rate=sample_rate
        )
        
        if not is_valid:
            return {"message": "Invalid audio signal detected"}
        
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