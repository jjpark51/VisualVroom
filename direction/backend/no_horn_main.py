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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=4):
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

# def process_audio(audio_file):
#     """Process the audio file and generate an image representation."""
    
#     # Convert to wav if necessary
#     wav_file = audio_file.replace('.m4a', '.wav')
#     convert_to_wav(audio_file, wav_file)
    
#     # Load and analyze audio file
#     y, sr = sf.read(wav_file)
#     duration = len(y) / sr
#     max_amplitude = float(abs(y).max())

#     logger.info(f"Received audio file: {os.path.basename(audio_file)}")
#     logger.info(f"Audio stats: channels={y.shape[1] if len(y.shape) > 1 else 1}, duration={duration:.2f}s, max_amplitude={max_amplitude:.2f}")

#     # Separate channels
#     top_mic = y[:, 0] if len(y.shape) > 1 else y
#     bottom_mic = y[:, 1] if len(y.shape) > 1 else y

def process_audio(wav_file):
    """Process the WAV file and generate an image representation."""
    # Load and analyze audio file
    y, sr = sf.read(wav_file)
    duration = len(y) / sr
    max_amplitude = float(abs(y).max())

    logger.info(f"Received audio file: {os.path.basename(wav_file)}")
    logger.info(f"Audio stats: channels={y.shape[1] if len(y.shape) > 1 else 1}, duration={duration:.2f}s, max_amplitude={max_amplitude:.2f}")
    # Separate channels
    top_mic = y[:, 0] if len(y.shape) > 1 else y
    bottom_mic = y[:, 1] if len(y.shape) > 1 else y
    # Convert to spectrograms and MFCCs
    def convert_to_spectrogram(audio, sr, n_fft=402, hop_length=201):
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        return librosa.amplitude_to_db(np.abs(D), ref=np.max)

    def convert_to_mfcc(audio, sr, n_mfcc=13, n_fft=402, hop_length=201):
        return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, fmax=sr/2)

    # Generate features
    top_spectrogram = convert_to_spectrogram(top_mic, sr)
    bottom_spectrogram = convert_to_spectrogram(bottom_mic, sr)
    top_mfcc = convert_to_mfcc(top_mic, sr)
    bottom_mfcc = convert_to_mfcc(bottom_mic, sr)

    # Convert arrays to images
    def array_to_image(array, width, height):
        array = ((array - array.min()) * (255.0 / (array.max() - array.min() + 1e-8))).astype(np.uint8)
        image = Image.fromarray(array).convert('L')
        return image.resize((width, height))

    # Convert to images with specific dimensions
    top_spectrogram_img = array_to_image(top_spectrogram, 241, 201)
    bottom_spectrogram_img = array_to_image(bottom_spectrogram, 241, 201)
    top_mfcc_img = array_to_image(top_mfcc, 241, 13)
    bottom_mfcc_img = array_to_image(bottom_mfcc, 241, 13)

    # Stitch images together
    final_img = Image.new('L', (241, 428))
    final_img.paste(top_mfcc_img, (0, 0))
    final_img.paste(top_spectrogram_img, (0, 13))
    final_img.paste(bottom_mfcc_img, (0, 214))
    final_img.paste(bottom_spectrogram_img, (0, 227))

    return final_img

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
    classes = ['Siren_L', 'Siren_R', 'Bike_L', 'Bike_R']
    predicted_class = classes[top_class.item()]
    vehicle_type, direction = predicted_class.split('_')
    confidence = float(top_prob.item())

    return vehicle_type, direction, confidence

# Load model globally to avoid reloading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "../checkpoints/no_horn_checkpoint.pth"
model = VisionTransformer(num_classes=4)
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

        # Process audio into image
        processed_image = process_audio(temp_wav)

        # Run inference
        vehicle_type, direction, confidence = predict_direction(model, processed_image, device)

        # Log inference
        logger.info(f"Prediction details - Vehicle: {vehicle_type}, Direction: {direction}, Confidence: {confidence:.4f}")

        return {
            "status": "success",
            "inference_result": {
                "vehicle_type": vehicle_type,
                "direction": direction,
                "confidence": round(confidence, 4),
                "should_notify": confidence > 0.97
            }
        }

    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
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


@app.post("/predict")
async def predict_audio(sample_rate: int = Form(...), 
                        left_channel: UploadFile = File(...),
                        right_channel: UploadFile = File(...)):
    try:
        # Read raw audio data from the request
        left_data = await left_channel.read()
        right_data = await right_channel.read()

        # Save the raw audio data to temporary files
        temp_left = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
        temp_right = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
        temp_left.write(left_data)
        temp_right.write(right_data)
        temp_left.close()
        temp_right.close()

        # Convert raw audio to WAV
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_wav.name, [sf.read(temp_left.name)[0], sf.read(temp_right.name)[0]], sample_rate)
        temp_wav.close()

        # Process audio into image
        processed_image = process_audio(temp_wav.name)

        # Run inference
        vehicle_type, direction, confidence = predict_direction(model, processed_image, device)

        # Log inference
        logger.info(f"Prediction details - Vehicle: {vehicle_type}, Direction: {direction}, Confidence: {confidence:.4f}")

        return {
            "status": "success",
            "inference_result": {
                "vehicle_type": vehicle_type,
                "direction": direction,
                "confidence": round(confidence, 4),
                "should_notify": confidence > 0.97
            }
        }

    except Exception as e:
        logger.error(f"Error processing audio data: {e}")
        return {"status": "error", "error": str(e)}

    finally:
        # Cleanup temporary files
        if os.path.exists(temp_left.name):
            os.unlink(temp_left.name)
        if os.path.exists(temp_right.name):
            os.unlink(temp_right.name)
        if os.path.exists(temp_wav.name):
            os.unlink(temp_wav.name)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
    # result = infer_static_audio("../test/Siren_L17.m4a")
 