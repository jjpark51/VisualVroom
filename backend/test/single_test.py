import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

def generate_spectrogram(y, sr, output_path):
    """Generate and save spectrogram image without axes or decorations"""
    plt.figure(frameon=False)
    
    # Create spectrogram using STFT
    D = librosa.stft(y, n_fft=402, hop_length=201)  # ~25ms window, ~12.5ms hop
    # Convert to power spectrogram
    D = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Create clean visualization
    ax = plt.axes([0, 0, 1, 1])  # Remove margins
    ax.set_axis_off()
    
    librosa.display.specshow(
        D,
        sr=sr,
        hop_length=201,
        y_axis='linear',
        cmap='gray',
        x_axis='time'
    )
    
    # Remove axes and margins
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    # Save without borders or padding
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    return D

def generate_mfcc(y, sr, output_path):
    """Generate and save MFCC image without axes or decorations"""
    # Calculate MFCC with specific parameters
    mfccs = librosa.feature.mfcc(
        y=y, 
        sr=sr,
        n_mfcc=13,  # Number of MFCCs to return
        n_fft=402,  # ~25ms window
        hop_length=201  # ~12.5ms hop
    )
    
    # Create clean visualization
    plt.figure(frameon=False)
    ax = plt.axes([0, 0, 1, 1])  # Remove margins
    ax.set_axis_off()
    
    librosa.display.specshow(
        mfccs,
        sr=sr,
        hop_length=201,
        x_axis='time',
        cmap='gray'
    )
    
    # Remove axes and margins
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    # Save without borders or padding
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    return mfccs

def process_single_audio_file(wav_path, output_dir):
    """Process a single audio file and generate visualization"""
    print(f"Processing audio file: {wav_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    y, sr = librosa.load(wav_path, sr=16000)
    
    # Base name for output files
    base_name = Path(wav_path).stem
    
    # Generate and save spectrogram
    spec_path = os.path.join(output_dir, f"{base_name}_spectrogram.png")
    generate_spectrogram(y, sr, spec_path)
    
    # Generate and save MFCC
    mfcc_path = os.path.join(output_dir, f"{base_name}_mfcc.png")
    generate_mfcc(y, sr, mfcc_path)
    
    # Create single channel stitched image
    channel_path = os.path.join(output_dir, f"{base_name}_channel.png")
    stitch_single_channel(mfcc_path, spec_path, channel_path)
    
    print(f"Finished processing audio file")
    return channel_path

def stitch_single_channel(mfcc_path, spec_path, output_path):
    """Stitch MFCC and spectrogram images for a single channel"""
    # Open both images
    mfcc_img = Image.open(mfcc_path)
    spec_img = Image.open(spec_path)
    
    # Convert to RGB if needed
    if mfcc_img.mode != 'RGB':
        mfcc_img = mfcc_img.convert('RGB')
    if spec_img.mode != 'RGB':
        spec_img = spec_img.convert('RGB')
    
    # Resize to match widths
    target_width = 241  # As per paper specifications
    target_height_mfcc = 13   # MFCC height (13 coefficients)
    target_height_spec = 201  # Spectrogram height
    
    mfcc_img = mfcc_img.resize((target_width, target_height_mfcc), Image.Resampling.LANCZOS)
    spec_img = spec_img.resize((target_width, target_height_spec), Image.Resampling.LANCZOS)
    
    # Create new image with combined height
    total_height = target_height_mfcc + target_height_spec
    stitched_img = Image.new('RGB', (target_width, total_height))
    
    # Paste images
    stitched_img.paste(mfcc_img, (0, 0))  # MFCC at top
    stitched_img.paste(spec_img, (0, target_height_mfcc))  # Spectrogram below
    
    # Save stitched image
    stitched_img.save(output_path, quality=95)
    return stitched_img

def main():
    # Define input and output directories
    input_wav = "./sounds/ambulance/sound_1.wav"  # Update this path to your audio file
    output_dir = "single_output"
    
    try:
        # Process single audio file
        final_path = process_single_audio_file(input_wav, output_dir)
        print(f"Final image saved to {final_path}")
        print("Processing completed successfully")
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()