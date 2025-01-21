import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

def generate_spectrogram(y, sr, output_path):
    """Generate and save spectrogram image without axes or decorations"""
    plt.figure(figsize=(10, 4))
    
    # Create spectrogram using STFT
    D = librosa.stft(y, n_fft=402, hop_length=201)  # ~25ms window, ~12.5ms hop
    # Convert to power spectrogram
    D = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Create clean visualization
    plt.figure(frameon=False)
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

def stitch_all_channels(left_path, rear_path, right_path, output_path):
    """Stitch three channel images vertically into a 241x642 image"""
    # Open all three images
    left_img = Image.open(left_path)
    rear_img = Image.open(rear_path)
    right_img = Image.open(right_path)
    
    # Ensure all images are in RGB mode
    if left_img.mode != 'RGB':
        left_img = left_img.convert('RGB')
    if rear_img.mode != 'RGB':
        rear_img = rear_img.convert('RGB')
    if right_img.mode != 'RGB':
        right_img = right_img.convert('RGB')
    
    # Create new image with target dimensions
    target_width = 241
    target_height = 642  # Total height for all three channels
    final_img = Image.new('RGB', (target_width, target_height))
    
    # Each channel image should be 214 pixels high (13 + 201)
    channel_height = 214
    
    # Resize each image if needed
    left_img = left_img.resize((target_width, channel_height), Image.Resampling.LANCZOS)
    rear_img = rear_img.resize((target_width, channel_height), Image.Resampling.LANCZOS)
    right_img = right_img.resize((target_width, channel_height), Image.Resampling.LANCZOS)
    
    # Paste images vertically
    final_img.paste(left_img, (0, 0))
    final_img.paste(rear_img, (0, channel_height))
    final_img.paste(right_img, (0, 2 * channel_height))
    
    # Save final stitched image
    final_img.save(output_path, quality=95)
    return final_img

def process_audio_files(left_wav, rear_wav, right_wav, output_dir):
    """Process three audio files and generate combined visualization"""
    print("Processing audio files...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each channel
    channels = {
        'left': left_wav,
        'rear': rear_wav,
        'right': right_wav
    }
    
    channel_paths = {}
    
    for position, wav_path in channels.items():
        print(f"Processing {position} channel...")
        base_name = Path(wav_path).stem
        
        # Load audio
        y, sr = librosa.load(wav_path, sr=16000)
        
        # Generate and save spectrogram
        spec_path = os.path.join(output_dir, f"{base_name}_spectrogram.png")
        generate_spectrogram(y, sr, spec_path)
        
        # Generate and save MFCC
        mfcc_path = os.path.join(output_dir, f"{base_name}_mfcc.png")
        generate_mfcc(y, sr, mfcc_path)
        
        # Create single channel stitched image
        channel_path = os.path.join(output_dir, f"{base_name}_channel.png")
        stitch_single_channel(mfcc_path, spec_path, channel_path)
        channel_paths[position] = channel_path
        
        print(f"Finished processing {position} channel")
    
    # Create final stitched image
    final_path = os.path.join(output_dir, "final_stitched.png")
    stitch_all_channels(
        channel_paths['left'],
        channel_paths['rear'],
        channel_paths['right'],
        final_path
    )
    print(f"Final stitched image saved to {final_path}")

def main():
    # Define input and output directories
    test_dir = "./sounds/ambulance"
    output_dir = "test_output"
    
    # Find WAV files for each channel
    wav_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
    
    if len(wav_files) < 3:
        print(f"Need at least 3 WAV files in {test_dir}")
        return
    
    # Sort files to ensure consistent ordering
    wav_files.sort()
    
    # Process the first three WAV files as left, rear, and right channels
    left_wav = os.path.join(test_dir, wav_files[0])
    rear_wav = os.path.join(test_dir, wav_files[1])
    right_wav = os.path.join(test_dir, wav_files[2])
    
    try:
        process_audio_files(left_wav, rear_wav, right_wav, output_dir)
        print("Processing completed successfully")
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()