import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import soundfile as sf
import pydub
from pydub import AudioSegment
import os

def convert_m4a_to_wav(m4a_path, wav_path):
    """Convert M4A file to WAV format."""
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(wav_path, format="wav")
    return wav_path

def split_channels(audio_path):
    """Split audio into left, right, and stereo channels."""
    audio = AudioSegment.from_file(audio_path)
    
    if audio.channels == 1:
        audio = audio.set_channels(2)
    
    samples = audio.get_array_of_samples()
    samples = np.array(samples).reshape((-1, audio.channels))
    
    left_channel = samples[:, 0].astype(np.float32) / 32768.0
    right_channel = samples[:, 1].astype(np.float32) / 32768.0
    stereo_channel = (left_channel + right_channel) / 2
    
    return left_channel, right_channel, stereo_channel

def apply_sliding_window(channel_data, sample_rate, window_size=3, step_size=1):
    """Apply sliding window to audio data and return both raw and amplified windows."""
    window_samples = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    
    n_windows = max(1, (len(channel_data) - window_samples) // step_samples + 1)
    
    raw_windows = []
    amplified_windows = []
    
    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        if end <= len(channel_data):
            raw_window = channel_data[start:end]
            
            # Create amplified version
            amplified_window = raw_window * 1.5
            amplified_window = np.clip(amplified_window, -1.0, 1.0)
            
            raw_windows.append(raw_window)
            amplified_windows.append(amplified_window)
    
    return raw_windows, amplified_windows

def generate_spectrogram(y, sr=44100):
    """Generate spectrogram from audio data."""
    D = librosa.stft(y, n_fft=402, hop_length=201)
    D = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', cmap='gray')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def generate_mfcc(y, sr=44100):
    """Generate MFCC image from audio data."""
    mfccs = librosa.feature.mfcc(
        y=y, 
        sr=sr,
        n_mfcc=13,
        n_fft=402,
        hop_length=201
    )
    
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='gray')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def stitch_channel_images(mfcc_img, spec_img):
    """Stitch MFCC and spectrogram images for a single channel."""
    target_width = 241
    mfcc_height = 13
    spec_height = 201
    
    mfcc_img = mfcc_img.resize((target_width, mfcc_height), Image.Resampling.LANCZOS)
    spec_img = spec_img.resize((target_width, spec_height), Image.Resampling.LANCZOS)
    
    combined = Image.new('RGB', (target_width, mfcc_height + spec_height))
    combined.paste(mfcc_img, (0, 0))
    combined.paste(spec_img, (0, mfcc_height))
    
    return combined

def stitch_all_channels(left_img, right_img, stereo_img):
    """Stitch all three channel images vertically."""
    target_width = 241
    channel_height = 214  # 13 + 201
    target_height = 642  # 3 * 214
    
    final_img = Image.new('RGB', (target_width, target_height))
    final_img.paste(left_img, (0, 0))
    final_img.paste(stereo_img, (0, channel_height))
    final_img.paste(right_img, (0, 2 * channel_height))
    
    return final_img

def save_audio_windows(windows, sample_rate, output_dir, prefix):
    """Save audio windows as WAV files."""
    for i, window in enumerate(windows):
        output_path = os.path.join(output_dir, f"{prefix}_window_{i+1}.wav")
        sf.write(output_path, window, sample_rate)

def process_audio_file(input_path, output_dir):
    """Process M4A file and create visualizations and audio files for both raw and amplified signals."""
    print("Processing audio file...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    amplified_dir = os.path.join(output_dir, "amplified")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(amplified_dir, exist_ok=True)
    
    # Create temporary WAV file
    temp_wav = "temp.wav"
    convert_m4a_to_wav(input_path, temp_wav)
    
    try:
        # Get sample rate
        audio = AudioSegment.from_file(temp_wav)
        sample_rate = audio.frame_rate
        
        # Split into channels
        left_channel, right_channel, stereo_channel = split_channels(temp_wav)
        
        # Process each channel with sliding windows
        channels = {
            'left': left_channel,
            'right': right_channel,
            'stereo': stereo_channel
        }
        
        for channel_name, channel_data in channels.items():
            raw_windows, amplified_windows = apply_sliding_window(channel_data, sample_rate)
            
            # Save audio windows
            save_audio_windows(raw_windows, sample_rate, 
                             os.path.join(raw_dir, channel_name), "raw")
            save_audio_windows(amplified_windows, sample_rate, 
                             os.path.join(amplified_dir, channel_name), "amplified")
            
            # Process raw windows
            for window_idx, window_data in enumerate(raw_windows):
                print(f"Processing raw {channel_name} window {window_idx + 1}/{len(raw_windows)}...")
                
                spec_img = generate_spectrogram(window_data, sr=sample_rate)
                mfcc_img = generate_mfcc(window_data, sr=sample_rate)
                stitched = stitch_channel_images(mfcc_img, spec_img)
                
                output_path = os.path.join(raw_dir, channel_name, 
                                         f"raw_visualization_window_{window_idx+1}.png")
                stitched.save(output_path, quality=95)
            
            # Process amplified windows
            for window_idx, window_data in enumerate(amplified_windows):
                print(f"Processing amplified {channel_name} window {window_idx + 1}/{len(amplified_windows)}...")
                
                spec_img = generate_spectrogram(window_data, sr=sample_rate)
                mfcc_img = generate_mfcc(window_data, sr=sample_rate)
                stitched = stitch_channel_images(mfcc_img, spec_img)
                
                output_path = os.path.join(amplified_dir, channel_name, 
                                         f"amplified_visualization_window_{window_idx+1}.png")
                stitched.save(output_path, quality=95)
        
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import soundfile as sf
import pydub
from pydub import AudioSegment
import os
from pathlib import Path
import glob

# [Previous helper functions remain the same until process_audio_file]


def get_test_number(filename):
    """Extract test number from filename (e.g., 'Test_3_B.m4a' -> '3')."""
    # Handle both 'Test' prefix cases
    if filename.startswith('Test_'):
        try:
            # Extract number between first '_' and second '_'
            return filename.split('_')[1]
        except (IndexError, ValueError):
            print(f"Warning: Unexpected filename format: {filename}")
            return '1'
    return '1'

def generate_output_filenames(test_number, direction, window_idx):
    """Generate filenames for raw and amplified outputs."""
    # Add window number suffix for multiple windows from same test
    window_suffix = f"_{window_idx+1}" if window_idx > 0 else ""
    return {
        'raw': f"test_{direction}_{test_number}{window_suffix}_raw.png",
        'amplified': f"test_{direction}_{test_number}{window_suffix}_amp.png"
    }

def process_audio_file(input_path, output_base_dir):
    """Process M4A file and create raw and amplified visualizations for each window."""
    print(f"\nProcessing file: {input_path}")
    
    # Get filename without extension and extract information
    input_path = Path(input_path)
    file_stem = input_path.stem
    direction = file_stem[-1]  # Get last character (L, B, or R)
    test_number = get_test_number(file_stem)
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, direction)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary WAV file
    temp_wav = f"temp_{file_stem}.wav"
    convert_m4a_to_wav(str(input_path), temp_wav)
    
    try:
        # Get sample rate
        audio = AudioSegment.from_file(temp_wav)
        sample_rate = audio.frame_rate
        
        # Split into channels
        left_channel, right_channel, stereo_channel = split_channels(temp_wav)
        
        # Process all channels with sliding windows
        left_raw, left_amp = apply_sliding_window(left_channel, sample_rate)
        right_raw, right_amp = apply_sliding_window(right_channel, sample_rate)
        stereo_raw, stereo_amp = apply_sliding_window(stereo_channel, sample_rate)
        
        # Process each window
        for window_idx in range(len(left_raw)):
            print(f"Processing window {window_idx + 1}/{len(left_raw)} for test {test_number}...")
            
            # Generate filenames for this window
            filenames = generate_output_filenames(test_number, direction, window_idx)
            
            # Process raw channels
            left_raw_spec = generate_spectrogram(left_raw[window_idx], sr=sample_rate)
            left_raw_mfcc = generate_mfcc(left_raw[window_idx], sr=sample_rate)
            left_raw_combined = stitch_channel_images(left_raw_mfcc, left_raw_spec)
            
            stereo_raw_spec = generate_spectrogram(stereo_raw[window_idx], sr=sample_rate)
            stereo_raw_mfcc = generate_mfcc(stereo_raw[window_idx], sr=sample_rate)
            stereo_raw_combined = stitch_channel_images(stereo_raw_mfcc, stereo_raw_spec)
            
            right_raw_spec = generate_spectrogram(right_raw[window_idx], sr=sample_rate)
            right_raw_mfcc = generate_mfcc(right_raw[window_idx], sr=sample_rate)
            right_raw_combined = stitch_channel_images(right_raw_mfcc, right_raw_spec)
            
            # Create raw stitched image
            final_raw = stitch_all_channels(left_raw_combined, right_raw_combined, stereo_raw_combined)
            raw_output_path = os.path.join(output_dir, filenames['raw'])
            final_raw.save(raw_output_path, quality=95)
            
            # Process amplified channels
            left_amp_spec = generate_spectrogram(left_amp[window_idx], sr=sample_rate)
            left_amp_mfcc = generate_mfcc(left_amp[window_idx], sr=sample_rate)
            left_amp_combined = stitch_channel_images(left_amp_mfcc, left_amp_spec)
            
            stereo_amp_spec = generate_spectrogram(stereo_amp[window_idx], sr=sample_rate)
            stereo_amp_mfcc = generate_mfcc(stereo_amp[window_idx], sr=sample_rate)
            stereo_amp_combined = stitch_channel_images(stereo_amp_mfcc, stereo_amp_spec)
            
            right_amp_spec = generate_spectrogram(right_amp[window_idx], sr=sample_rate)
            right_amp_mfcc = generate_mfcc(right_amp[window_idx], sr=sample_rate)
            right_amp_combined = stitch_channel_images(right_amp_mfcc, right_amp_spec)
            
            # Create amplified stitched image
            final_amp = stitch_all_channels(left_amp_combined, right_amp_combined, stereo_amp_combined)
            amp_output_path = os.path.join(output_dir, filenames['amplified'])
            final_amp.save(amp_output_path, quality=95)
            
        print(f"Completed processing {input_path}")
        
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
def process_directory(input_base_dir, output_base_dir):
    """Process all M4A files in the directory structure."""
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create direction directories
    for direction in ['L', 'B', 'R']:
        os.makedirs(os.path.join(output_base_dir, direction), exist_ok=True)
    
    # Get all subdirectories
    subdirs = ['Bike_B', 'Bike_L', 'Bike_R']
    
    total_files = 0
    processed_files = 0
    
    # Count total files first
    for subdir in subdirs:
        input_dir = os.path.join(input_base_dir, subdir)
        if os.path.exists(input_dir):
            total_files += len(glob.glob(os.path.join(input_dir, "*.m4a")))
    
    print(f"Found {total_files} files to process")
    
    # Process each subdirectory
    for subdir in subdirs:
        input_dir = os.path.join(input_base_dir, subdir)
        if not os.path.exists(input_dir):
            print(f"Directory not found: {input_dir}")
            continue
        
        # Process all M4A files in the subdirectory
        for input_file in glob.glob(os.path.join(input_dir, "*.m4a")):
            try:
                process_audio_file(input_file, output_base_dir)
                processed_files += 1
                print(f"Progress: {processed_files}/{total_files} files processed")
            except Exception as e:
                print(f"Error processing {input_file}: {str(e)}")
    
    print(f"\nProcessing complete. Successfully processed {processed_files}/{total_files} files.")

def main():
    input_base_dir = "custom_dataset"
    output_base_dir = "processed_dataset"
    
    if not os.path.exists(input_base_dir):
        print(f"Error: Input directory '{input_base_dir}' not found")
        return
    
    try:
        process_directory(input_base_dir, output_base_dir)
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()