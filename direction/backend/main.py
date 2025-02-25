import torch
from torchvision import transforms
from PIL import Image
import soundfile as sf
from pydub import AudioSegment
import librosa
import numpy as np
import torch.nn as nn
from torchvision.models import vit_b_16
import logging
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from datetime import datetime
import whisper
from typing import Optional
import tempfile
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

whisper_model = whisper.load_model("small")


@app.post("/transcribe")
async def transcribe_audio(
    sample_rate: int = Form(...),
    audio_data: UploadFile = File(...)
):
    """
    Endpoint for speech-to-text transcription using Whisper AI.
    Processes complete audio recordings and returns transcription.
    """
    temp_file = None
    try:
        # Read raw audio data
        content = await audio_data.read()
        
        # Convert raw PCM data to numpy array
        audio_np = np.frombuffer(content, dtype=np.int16)
        audio_float = audio_np.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

        # Save as temporary WAV file (Whisper requires a file)
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio_float, sample_rate, format='WAV')

        # Log audio details
        logger.info(f"Processing audio file:")
        logger.info(f"Sample rate: {sample_rate} Hz")
        logger.info(f"Duration: {len(audio_float) / sample_rate:.2f} seconds")
        logger.info(f"Max amplitude: {np.max(np.abs(audio_float)):.4f}")

        # Transcribe using Whisper
        result = whisper_model.transcribe(
            temp_file.name,
            language="en",
            fp16=False
        )

        transcribed_text = result["text"].strip()
        logger.info(f"Transcription result: {transcribed_text}")

        return {
            "status": "success",
            "text": transcribed_text
        }

    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

    finally:
        # Cleanup
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=6):
        super(VisionTransformer, self).__init__()
        self.vit = vit_b_16()
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_features, num_classes)
        self.vit.conv_proj = nn.Conv2d(1, self.vit.conv_proj.out_channels, kernel_size=16, stride=16)

    def forward(self, x):
        return self.vit(x)

def convert_to_wav(audio_file, output_file):
    audio = AudioSegment.from_file(audio_file)
    audio.export(output_file, format="wav")
    return output_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler('app.log')  # Also save to file
    ]
)
logger = logging.getLogger(__name__)

# Make sure all loggers are capturing INFO level
logging.getLogger().setLevel(logging.INFO)

# Add a print statement as backup
def log_with_print(message):
    print(message)  # Backup print
    logger.info(message)  # Normal logging

def process_audio(wav_file):
    """Process the WAV file and generate an image representation."""
    # Load and analyze audio file
    y, sr = sf.read(wav_file)
    duration = len(y) / sr
    
    # Separate channels and log their stats
    top_mic = y[:, 0] if len(y.shape) > 1 else y
    bottom_mic = y[:, 1] if len(y.shape) > 1 else y
    
    # Calculate amplitude-based direction
    top_max = float(abs(top_mic).max())
    bottom_max = float(abs(bottom_mic).max())
    amplitude_direction = "L" if top_max > bottom_max else "R"
    amplitude_ratio = top_max / bottom_max if top_max > bottom_max else bottom_max / top_max
    
    # Log detailed channel information with print backup
    log_with_print(f"\n{'='*50}")
    log_with_print(f"Processing file: {os.path.basename(wav_file)}")
    log_with_print(f"Audio stats: channels={y.shape[1] if len(y.shape) > 1 else 1}, duration={duration:.2f}s")
    log_with_print(f"Top mic (left channel) stats:")
    log_with_print(f"  - max_amplitude={top_max:.4f}")
    log_with_print(f"  - mean_amplitude={float(abs(top_mic).mean()):.4f}")
    log_with_print(f"  - rms={float(np.sqrt(np.mean(top_mic**2))):.4f}")
    
    log_with_print(f"Bottom mic (right channel) stats:")
    log_with_print(f"  - max_amplitude={bottom_max:.4f}")
    log_with_print(f"  - mean_amplitude={float(abs(bottom_mic).mean()):.4f}")
    log_with_print(f"  - rms={float(np.sqrt(np.mean(bottom_mic**2))):.4f}")

    # Generate features
    logger.info("Generating spectrograms and MFCCs...")
    
    # Convert to spectrograms and MFCCs with additional stats
    def convert_to_spectrogram(audio, sr, n_fft=402, hop_length=201):
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        spec_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        log_with_print(f"Spectrogram stats: shape={spec_db.shape}, range=[{spec_db.min():.2f}, {spec_db.max():.2f}]")
        return spec_db

    def convert_to_mfcc(audio, sr, n_mfcc=13, n_fft=402, hop_length=201):
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, 
                                  hop_length=hop_length, fmax=sr/2)
        log_with_print(f"MFCC stats: shape={mfcc.shape}, range=[{mfcc.min():.2f}, {mfcc.max():.2f}]")
        return mfcc

    top_spectrogram = convert_to_spectrogram(top_mic, sr)
    bottom_spectrogram = convert_to_spectrogram(bottom_mic, sr)
    top_mfcc = convert_to_mfcc(top_mic, sr)
    bottom_mfcc = convert_to_mfcc(bottom_mic, sr)

    # Convert arrays to images
    def array_to_image(array, width, height, name):
        array = ((array - array.min()) * (255.0 / (array.max() - array.min() + 1e-8))).astype(np.uint8)
        image = Image.fromarray(array).convert('L')
        resized = image.resize((width, height))
        logger.info(f"{name} image size: {resized.size}, mode: {resized.mode}")
        return resized

    # Convert to images with specific dimensions
    top_spectrogram_img = array_to_image(top_spectrogram, 241, 201, "Top spectrogram")
    bottom_spectrogram_img = array_to_image(bottom_spectrogram, 241, 201, "Bottom spectrogram")
    top_mfcc_img = array_to_image(top_mfcc, 241, 13, "Top MFCC")
    bottom_mfcc_img = array_to_image(bottom_mfcc, 241, 13, "Bottom MFCC")

    # Stitch images together
    final_img = Image.new('L', (241, 428))
    final_img.paste(top_mfcc_img, (0, 0))
    final_img.paste(top_spectrogram_img, (0, 13))
    final_img.paste(bottom_mfcc_img, (0, 214))
    final_img.paste(bottom_spectrogram_img, (0, 227))

    # Save the stitched image for inspection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "debug_images"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save individual components
    top_spectrogram_img.save(f"{save_dir}/top_spec_{timestamp}.png")
    bottom_spectrogram_img.save(f"{save_dir}/bottom_spec_{timestamp}.png")
    top_mfcc_img.save(f"{save_dir}/top_mfcc_{timestamp}.png")
    bottom_mfcc_img.save(f"{save_dir}/bottom_mfcc_{timestamp}.png")
    final_img.save(f"{save_dir}/stitched_{timestamp}.png")
    
    log_with_print(f"Processing completed successfully")
    log_with_print(f"{'='*50}\n")
    
    return final_img, amplitude_direction, amplitude_ratio

def predict_direction(model, image, device):
    """Run inference and return predicted class and confidence score."""
    
    # Apply identical transformation as standalone script
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # Transform image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Ensure model is in eval mode
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probabilities, 1)

    # Class mapping
    classes = [
        'Siren_L', 'Siren_R', 'Bike_L', 'Bike_R', 'Horn_L', 'Horn_R'
    ]    
    predicted_class = classes[top_class.item()]
    vehicle_type, direction = predicted_class.split('_')
    confidence = float(top_prob.item())

    return vehicle_type, direction, confidence

# Load model globally to avoid reloading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "./checkpoints/feb_25_checkpoint.pth"
model = VisionTransformer(num_classes=6)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


@app.post("/test")
async def test_audio(audio_file: UploadFile = File(...)):
    """Endpoint for testing audio inference."""
    temp_m4a = tempfile.NamedTemporaryFile(suffix='.m4a', delete=False)
    temp_wav = None
    
    try:
        logger.info(f"Received audio file: {audio_file.filename}")

        # Save the file temporarily
        content = await audio_file.read()
        temp_m4a.write(content)
        temp_m4a.close()

        # Convert to WAV
        temp_wav = temp_m4a.name.replace('.m4a', '.wav')
        convert_to_wav(temp_m4a.name, temp_wav)

        # Process audio into image and get amplitude-based direction
        processed_image, amplitude_direction, amplitude_ratio = process_audio(temp_wav)

        # Run model inference
        vehicle_type, model_direction, confidence = predict_direction(model, processed_image, device)

        # Use amplitude-based direction instead of model prediction
        final_direction = amplitude_direction

        # Log inference results
        logger.info(f"\nPrediction Results:")
        logger.info(f"Model prediction: {vehicle_type}_{model_direction}")
        # logger.info(f"Amplitude-based direction: {amplitude_direction}")
        logger.info(f"Final decision: {vehicle_type}_{final_direction}")
        logger.info(f"Amplitude ratio: {amplitude_ratio:.4f}")
        logger.info(f"Model confidence: {confidence:.4f}")

        return {
            "status": "success",
            "inference_result": {
                "vehicle_type": vehicle_type,
                "direction": model_direction,
                "confidence": round(confidence, 4),
                "should_notify": confidence > 0.97,
                "amplitude_ratio": round(amplitude_ratio, 4)
            }
        }

    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        return {"status": "error", "error": str(e)}
    
    finally:
        if temp_m4a and os.path.exists(temp_m4a.name):
            os.unlink(temp_m4a.name)
        if temp_wav and os.path.exists(temp_wav):
            os.unlink(temp_wav)

def infer_static_audio(audio_file_path):
    """Run inference on a static audio file."""
    
    # Check if file exists
    if not os.path.exists(audio_file_path):
        logger.error(f"File not found: {audio_file_path}")
        return {"status": "error", "message": "File not found"}

    try:
        logger.info(f"Processing static audio file: {audio_file_path}")

        # Convert to WAV
        temp_wav = audio_file_path.replace('.m4a', '.wav')
        convert_to_wav(audio_file_path, temp_wav)

        # Process audio into spectrogram & MFCC image
        processed_image = process_audio(temp_wav)

        # Run inference
        vehicle_type, direction, confidence = predict_direction(model, processed_image, device)

        # Log and return results
        logger.info(f"Prediction details - Vehicle: {vehicle_type}, Direction: {direction}, Confidence: {confidence:.4f}")

        return {
            "status": "success",
            "file": os.path.basename(audio_file_path),
            "inference_result": {
                "vehicle_type": vehicle_type,
                "direction": direction,
                "confidence": round(confidence, 4),
                "should_notify": confidence > 0.97
            }
        }

    except Exception as e:
        logger.error(f"Error during static file inference: {e}")
        return {"status": "error", "message": str(e)}
    
    finally:
        # Cleanup WAV file
        if os.path.exists(temp_wav):
            os.unlink(temp_wav)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
