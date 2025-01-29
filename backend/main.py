from flask import Flask, request, jsonify
import numpy as np
import wave
import os
from datetime import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

#Configure upload folder
UPLOAD_FOLDER = 'recordings'
IMAGE_FOLDER = 'images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


@app.route('/', methods=['GET'])
def hello_world():
    print("Hello World")

def save_wav(data, filepath, channels=1):
    """Save raw audio data as WAV file"""
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(channels)  # 1 for mono, 2 for stereo
        wav_file.setsampwidth(2)         # 2 bytes per sample (16-bit)
        wav_file.setframerate(16000)     # 16 kHz sample rate
        wav_file.writeframes(data)



def generate_spectrogram(audio_path, output_path):
    """Generate and save spectrogram image"""
    # Load audio file
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Create spectrogram
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # Save and close
    plt.savefig(output_path)
    plt.close()

def generate_mfcc(audio_path, output_path):
    """Generate and save MFCC image"""
    # Load audio file
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Calculate MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Create MFCC image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar(format='%+2.0f')
    plt.title('MFCC')
    
    # Save and close
    plt.savefig(output_path)
    plt.close()

@app.route('/upload', methods=['POST'])
def upload_audio():
    try:
        # Check if all required files are present
        if 'left_mic' not in request.files or \
           'right_mic' not in request.files or \
           'both_mics' not in request.files:
            return jsonify({'error': 'Missing required audio files'}), 400

        # Get current timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        session_dir = os.path.join(UPLOAD_FOLDER, timestamp)
        image_dir = os.path.join(IMAGE_FOLDER, timestamp)
        os.makedirs(session_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        
        # Process each channel
        channels = {
            'left': request.files['left_mic'].read(),
            'right': request.files['right_mic'].read(),
            'both': request.files['both_mics'].read()
        }
        
        file_paths = {}
        for channel_name, data in channels.items():
            # Save WAV file
            wav_path = os.path.join(session_dir, f'{channel_name}_channel.wav')
            save_wav(data, wav_path, channels=2 if channel_name == 'both' else 1)
            
            # Generate and save spectrogram
            spec_path = os.path.join(image_dir, f'{channel_name}_spectrogram.png')
            generate_spectrogram(wav_path, spec_path)
            
            # Generate and save MFCC
            mfcc_path = os.path.join(image_dir, f'{channel_name}_mfcc.png')
            generate_mfcc(wav_path, mfcc_path)
            
            file_paths[channel_name] = {
                'wav': f'{channel_name}_channel.wav',
                'spectrogram': f'{channel_name}_spectrogram.png',
                'mfcc': f'{channel_name}_mfcc.png'
            }
        
        return jsonify({
            'message': 'Audio files processed and images generated successfully',
            'session_id': timestamp,
            'files': file_paths
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to process audio files',
            'details': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=8888, debug=True)