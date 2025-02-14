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
import whisper


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectionalSoundViT(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_features, num_classes)
        self.vit.encoder.gradient_checkpointing = True

    def forward(self, x):
        return self.vit(x)

class ModelService:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.model = DirectionalSoundViT(num_classes=8).to(self.device)
        self.load_model(model_path)
        self.model.eval()
        
        self.class_names = [
            'Ambulance Left',  'Ambulance Right',
            'Car_Horn Left',  'Car_Horn Right',
            'Fire_Truck Left', 'Fire_Truck Right',
            'Police_Car Left',  'Police_Car Right',
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.energy_threshold = 0.001
        self.noise_threshold_db = 0
        self.min_confidence = 0.8
        self.high_confidence = 0.9
        self.prediction_history = []
        self.history_size = 3
    
    def check_signal_quality(self, audio: np.ndarray) -> bool:
        energy = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        crest_factor = peak / (np.sqrt(np.mean(audio**2)) + 1e-10)
        
        if energy < self.energy_threshold:
            return False
            
        if crest_factor > 30:
            return False
        
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)
        if zero_crossings > 0.9:
            return False
        
        return True
    
    def analyze_spectral_content(self, D: np.ndarray) -> bool:
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        mean_magnitude = np.mean(D_db)
        std_magnitude = np.std(D_db)
        max_magnitude = np.max(D_db)
        
        if mean_magnitude < -70:
            return False
            
        if std_magnitude < 3:
            return False
            
        if max_magnitude < -40:
            return False
        
        return True

    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info("Model loaded successfully")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def process_audio(self, audio_data: List[bytes], sample_rate: int = 16000) -> tuple[Image.Image, bool]:
        images = []
        channels_valid = []
        
        for i, channel_data in enumerate(audio_data):
            try:
                audio = np.frombuffer(channel_data, dtype=np.int16).astype(float) / 32768.0
                signal_valid = self.check_signal_quality(audio)
                channels_valid.append(signal_valid)
                
                D = librosa.stft(audio, n_fft=402, hop_length=201)
                spectral_valid = self.analyze_spectral_content(D)
                channels_valid[-1] &= spectral_valid
                
                if signal_valid or spectral_valid:
                    D = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                    
                    mfccs = librosa.feature.mfcc(
                        y=audio,
                        sr=sample_rate,
                        n_mfcc=13,
                        n_fft=402,
                        hop_length=201
                    )
                    
                    spec_img = self.array_to_image(D)
                    mfcc_img = self.array_to_image(mfccs)
                    channel_img = self.stitch_channel_images(mfcc_img, spec_img)
                    images.append(channel_img)
                
            except Exception as e:
                channels_valid.append(False)
                continue
        
        is_valid_signal = any(channels_valid)
        
        if not is_valid_signal or len(images) == 0:
            return None, False
            
        return self.stitch_all_channels(images), True
    
    def check_temporal_consistency(self, prediction: dict) -> bool:
        if not self.prediction_history:
            self.prediction_history.append(prediction)
            return True
            
        recent_predictions = self.prediction_history[-self.history_size:]
        matching_predictions = sum(
            1 for p in recent_predictions
            if p['vehicle_type'] == prediction['vehicle_type'] and 
            p['direction'] == prediction['direction']
        )
        
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
            
        return matching_predictions >= (len(recent_predictions) // 2)
    
    def array_to_image(self, array: np.ndarray) -> Image.Image:
        array = ((array - array.min()) * (255.0 / (array.max() - array.min()))).astype(np.uint8)
        return Image.fromarray(array).convert('RGB')

    def stitch_channel_images(self, mfcc_img: Image.Image, spec_img: Image.Image) -> Image.Image:
        target_width = 241
        mfcc_height = 13
        spec_height = 201
        
        mfcc_img = mfcc_img.resize((target_width, mfcc_height))
        spec_img = spec_img.resize((target_width, spec_height))
        
        combined = Image.new('RGB', (target_width, mfcc_height + spec_height))
        combined.paste(mfcc_img, (0, 0))
        combined.paste(spec_img, (0, mfcc_height))
        
        return combined

    def stitch_all_channels(self, channel_images: List[Image.Image]) -> Image.Image:
        if not channel_images:
            raise ValueError("No channel images provided")
            
        channel_height = 214
        total_height = channel_height * len(channel_images)
        width = channel_images[0].width
        
        final_img = Image.new('RGB', (width, total_height))
        for i, img in enumerate(channel_images):
            final_img.paste(img, (0, i * channel_height))
        
        return final_img

    def predict(self, image: Image.Image):
        try:
            if image is None:
                return None
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                k = 2
                top_probs, top_indices = torch.topk(probabilities, k)
                
                confidence_margin = top_probs[0][0] - top_probs[0][1]
                predicted_class = top_indices[0][0].item()
                confidence = top_probs[0][0].item()
                
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
                    
                    if confidence > self.high_confidence or self.check_temporal_consistency(prediction):
                        if confidence > 0.98:
                            print(prediction)
                        return prediction
                        
                return None
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Initialize FastAPI app
app = FastAPI(title="Vehicle Sound Detection Model Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = None

@app.on_event("startup")
async def startup_event():
    global model_service, whisper_model
    try:
        model_service = ModelService("eight_class/short_best_model.pth")
        whisper_model = whisper.load_model("small")
        logger.info("Model services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model services: {str(e)}")
        raise
@app.post("/transcribe")
async def transcribe_audio(
    audio_data: UploadFile = File(...),
    sample_rate: int = Form(16000)
):
    try:
        # Read the audio data
        audio_bytes = await audio_data.read()
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Process audio for Whisper
        audio = whisper.pad_or_trim(audio_np)
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
        
        # Detect language
        _, probs = whisper_model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        
        # Transcribe
        options = whisper.DecodingOptions(
            task="transcribe",
            language=detected_lang,
            fp16=torch.cuda.is_available()
        )
        result = whisper.decode(whisper_model, mel, options)
        
        return {
            "text": result.text,
            "language": detected_lang,
            "confidence": float(max(probs.values()))
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(
    sample_rate: int = Form(16000),
    left_channel: UploadFile = File(...),
    right_channel: UploadFile = File(...),
    rear_channel: UploadFile = File(...)
):
    try:
        audio_data = [
            await left_channel.read(),
            await right_channel.read(),
            await rear_channel.read()
        ]
        
        processed_image, is_valid = model_service.process_audio(
            audio_data, 
            sample_rate=sample_rate
        )
        
        if not is_valid:
            return {"message": "Invalid audio signal detected"}
        
        result = model_service.predict(processed_image)
        
        if result is None:
            return {"message": "No confident prediction"}
            
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)