
# VisualVroom Backend

## Overview
This is the backend server for VisualVroom, an assistive technology application that helps identify vehicle alert sounds and their directions. The system uses a Vision Transformer (ViT) model to process audio spectrograms and provides real-time classification through WebSocket connections.

## Features
- Real-time audio processing pipeline
- WebSocket server for streaming audio data
- Vision Transformer (ViT) model for sound classification
- Multi-channel audio support (Left, Right, Rear)
- Spectrogram and MFCC feature extraction
- Edge AI optimization
- High-accuracy vehicle sound classification

## Technical Stack
- Python 3.8+
- PyTorch for deep learning
- Librosa for audio processing
- Websockets for real-time communication
- Numpy for numerical operations
- Matplotlib for visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jjpark51/VisualVroom-Android.git
cd VisualVroom-Android/backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install torch torchvision torchaudio
pip install websockets numpy librosa matplotlib
```

## Project Structure
```
backend/
├── model_checkpoints/      # Trained model checkpoints
├── websocket_server.py     # WebSocket server implementation
├── vit.ipynb              # Vision Transformer model definition
├── inference.py           # Model inference code
└── README.md
```

## Model Architecture
The system uses a Vision Transformer (ViT) model adapted for audio classification:
- Input: Stitched spectrograms and MFCC features
- Architecture: ViT-B/16 backbone
- Output: 12 classes (4 vehicle types × 3 directions)
- Preprocessing: Audio to spectrogram conversion with MFCC feature extraction

## Running the Server

1. Start the WebSocket server:
```bash
python websocket_server.py
```

The server will start listening on `ws://0.0.0.0:8080`

## API Reference

### WebSocket Endpoints

#### Audio Stream Endpoint
- URL: `ws://server:8080`
- Protocol: WebSocket
- Message Format: Binary audio data with channel identifier

Input Format:
- First byte: Channel identifier (0: Left, 1: Right, 2: Rear)
- Remaining bytes: Raw audio data (16kHz, 16-bit PCM)

Response Format:
```json
{
    "vehicle_type": "string",
    "direction": "string",
    "confidence": float,
    "timestamp": float,
    "should_notify": boolean
}
```

## Model Training

To train the model:

1. Prepare your dataset in the required format
2. Run the training script:
```bash
python vit.py
```

Training parameters can be modified in the script:
- Batch size: 32
- Learning rate: 1e-4
- Number of epochs: 30
- Early stopping patience: 7

## Performance

Current model performance metrics:
- Accuracy: 84.38% (Server-based)
- Recall: 84.38%
- Precision: 85.44%
- F1 Score: 84.66%

Edge computing performance:
- Accuracy: 83.01%
- Average inference time: 8.75ms

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

## Monitoring and Logging

The server provides detailed logging for:
- WebSocket connections/disconnections
- Model inference results
- Error handling
- Performance metrics

## Error Handling

The system includes robust error handling for:
- Audio processing failures
- Model inference issues
- Connection problems
- Invalid data formats

## License
[TBD]



