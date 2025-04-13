# VisualVroom - ML Backend Server

## Overview
The backend server component of the VisualVroom system built with PyTorch and FastAPI. 

## Features
- **Audio Processing Pipeline**: Transforms raw audio into spectrograms and MFCCs
- **Vehicle Sound Classification**: Identifies sirens, bicycle bells, and car horns
- **Direction Detection**: Determines if sounds are coming from the left or right
- **Confidence Scoring**: Provides confidence levels for predictions
- **Speech-to-Text API**: Offers Whisper AI integration for speech transcription

## Technical Architecture
- **ML Model**: Vision Transformer (ViT-B/16) adapted for audio spectrogram classification
- **Audio Processing**: Uses librosa and soundfile for feature extraction
- **API Framework**: FastAPI for efficient, asynchronous endpoints
- **Speech Recognition**: Uses OpenAI's Whisper model for transcription

## Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU (recommended for production)
- 2GB+ RAM

## Installation

### Manual Installation
```bash
# Clone repository
git clone https://github.com/jjpark51/visualvroom-backend.git
cd visualvroom-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

# Download the model checkpoint
mkdir -p checkpoints
wget -O checkpoints/feb_25_checkpoint.pth https://[your-model-host]/feb_25_checkpoint.pth

# Start the server
python main.py
```

## API Endpoints

### 1. `/test` (POST)
Process audio files for vehicle sound detection.

**Parameters:**
- `audio_file`: M4A audio file (will be converted to WAV)

**Response:**
```json
{
  "status": "success",
  "inference_result": {
    "vehicle_type": "Siren|Bike|Horn",
    "direction": "L|R",
    "confidence": 0.97,
    "should_notify": true,
    "amplitude_ratio": 1.2,
    "too_quiet": false
  }
}
```

### 2. `/transcribe` (POST)
Transcribe speech to text using Whisper AI.

**Parameters:**
- `sample_rate`: Integer sample rate in Hz
- `audio_data`: Raw PCM audio data

**Response:**
```json
{
  "status": "success",
  "text": "Transcribed text from the audio"
}
```

## Model Architecture
The system uses a Vision Transformer (ViT) architecture adapted for audio processing:
- Input: Composite images containing both MFCC and spectrogram representations
- Architecture: Modified ViT-B/16 with a custom input layer for grayscale images
- Output: 6 classes (Siren_L, Siren_R, Bike_L, Bike_R, Horn_L, Horn_R)

## Audio Processing Pipeline
1. **Preprocessing**: Audio is converted to WAV format and analyzed for amplitude
2. **Feature Extraction**:
   - Spectrograms are generated for each channel
   - MFCCs are extracted for additional features
3. **Image Generation**: Features are converted to grayscale images and stitched together
4. **Model Inference**: The composite image is fed into the Vision Transformer
5. **Post-processing**: Additional amplitude-based direction analysis is performed as a verification step



