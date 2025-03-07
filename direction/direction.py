import soundfile as sf
import numpy as np
from pydub import AudioSegment
import librosa
from PIL import Image
import os
import glob

def convert_to_wav(audio_file, output_file):
    # Convert audio file to .wav format
    audio = AudioSegment.from_file(audio_file)
    audio.export(output_file, format="wav")

def separate_interview_audio(audio_file):
    # Load the stereo audio file
    y, sr = sf.read(audio_file)
    
    # y will be a 2D array where:
    # y[:, 0] = top microphone (left channel)
    # y[:, 1] = bottom microphone (right channel)
    
    top_mic = y[:, 0]
    bottom_mic = y[:, 1]
    
    # Save separated channels
    sf.write('top_mic.wav', top_mic, sr)
    sf.write('bottom_mic.wav', bottom_mic, sr)
    
    return top_mic, bottom_mic, sr

def calculate_sound_direction(top_mic, bottom_mic):
    # Calculate the energy of each channel
    top_energy = np.sum(np.square(top_mic))
    bottom_energy = np.sum(np.square(bottom_mic))
    
    # Determine the direction based on the energy comparison
    if top_energy > bottom_energy:
        return "left"
    else:
        return "right"

def segment_audio(audio, sr, window_size=3, step_size=1):
    window_samples = window_size * sr
    step_samples = step_size * sr
    segments = []
    
    for start in range(0, len(audio) - window_samples + 1, step_samples):
        end = start + window_samples
        segments.append(audio[start:end])
    
    return segments

def augment_audio(audio, factors=[1.5, 1.3]):
    augmented_audios = [audio * factor for factor in factors]
    return augmented_audios

def convert_to_spectrogram(audio, sr, n_fft=402, hop_length=201):
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return D_db

def convert_to_mfcc(audio, sr, n_mfcc=13, n_fft=402, hop_length=201):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs

def array_to_image(array, width, height):
    array = ((array - array.min()) * (255.0 / (array.max() - array.min()))).astype(np.uint8)
    image = Image.fromarray(array).convert('L')
    image = image.resize((width, height))
    return image

def process_audio_file(audio_file):
    # Convert to .wav format
    wav_file_path = audio_file.replace('.m4a', '.wav')
    convert_to_wav(audio_file, wav_file_path)
    
    # Separate interview audio
    top_mic, bottom_mic, sr = separate_interview_audio(wav_file_path)
    
    # Segment audio
    top_segments = segment_audio(top_mic, sr)
    bottom_segments = segment_audio(bottom_mic, sr)
    
    all_segments = []
    
    for top_seg, bottom_seg in zip(top_segments, bottom_segments):
        # Augment audio
        top_augmented = augment_audio(top_seg)
        bottom_augmented = augment_audio(bottom_seg)
        
        # Combine raw and augmented data
        all_segments.append((top_seg, bottom_seg))
        all_segments.extend(zip(top_augmented, bottom_augmented))
    
    return all_segments, sr

def stitch_images(top_mfcc_img, top_spectrogram_img, bottom_mfcc_img, bottom_spectrogram_img):
    width = 241
    height = 428
    
    final_img = Image.new('L', (width, height))
    
    final_img.paste(top_mfcc_img, (0, 0))
    final_img.paste(top_spectrogram_img, (0, 13))
    final_img.paste(bottom_mfcc_img, (0, 214))
    final_img.paste(bottom_spectrogram_img, (0, 227))
    
    return final_img

def generate_images(segments, sr):
    images = []
    for top, bottom in segments:
        # Convert to spectrogram and MFCC
        top_spectrogram = convert_to_spectrogram(top, sr)
        bottom_spectrogram = convert_to_spectrogram(bottom, sr)
        top_mfcc = convert_to_mfcc(top, sr)
        bottom_mfcc = convert_to_mfcc(bottom, sr)
        
        # Convert to images
        top_spectrogram_img = array_to_image(top_spectrogram, 241, 201)
        bottom_spectrogram_img = array_to_image(bottom_spectrogram, 241, 201)
        top_mfcc_img = array_to_image(top_mfcc, 241, 13)
        bottom_mfcc_img = array_to_image(bottom_mfcc, 241, 13)
        
        # Stitch images
        stitched_img = stitch_images(top_mfcc_img, top_spectrogram_img, bottom_mfcc_img, bottom_spectrogram_img)
        images.append(stitched_img)
    
    return images

def save_images(images, output_dir, source_file, start_index=0):
    """
    Save images with a prefix based on the containing folder and original source file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get folder name to use as prefix
    folder_name = os.path.basename(output_dir)
    
    # Get source filename without extension
    source_name = os.path.splitext(os.path.basename(source_file))[0]
    
    for i, img in enumerate(images):
        # Use folder name and source file as prefix
        img_path = os.path.join(output_dir, f"{folder_name}_{source_name}_{start_index + i}.png")
        img.save(img_path)
        print(f"Saved image: {img_path}")

def process_directory(input_dir, file_pattern, output_dir):
    file_paths = glob.glob(os.path.join(input_dir, file_pattern))
    start_index = 0
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        # Process audio file
        segments, sample_rate = process_audio_file(file_path)
        
        # Generate images
        images = generate_images(segments, sample_rate)
        
        # Save images with source filename info
        save_images(images, output_dir, file_path, start_index)
        
        # Update start index for the next file
        start_index += len(images)
        
        # Calculate sound direction for each segment
        for i, (top, bottom) in enumerate(segments):
            direction = calculate_sound_direction(top, bottom)
            print(f"Segment {i}: The sound is coming from the {direction}.")

def create_train_val_test_splits(base_dir, train_ratio=0.7, val_ratio=0.1):
    """
    Create proper train/validation/test splits based on audio source files,
    not on individual generated image files.
    
    This ensures no data leakage between splits.
    """
    import shutil
    import random
    
    # Create output directories
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")
    
    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) 
                 and d not in ["train", "val", "test"]]
    
    for class_name in class_dirs:
        class_path = os.path.join(base_dir, class_name)
        
        # Create class directories in train/val/test
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        for directory in [train_class_dir, val_class_dir, test_class_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Group files by their source audio file
        files = glob.glob(os.path.join(class_path, "*.png"))
        
        # Extract source names from filenames
        source_files = {}
        for file in files:
            filename = os.path.basename(file)
            # Format should be class_sourcename_index.png
            parts = filename.split('_')
            if len(parts) >= 3:
                # Join all parts except the last one (index) and the first one (class)
                source_name = '_'.join(parts[1:-1])
                if source_name not in source_files:
                    source_files[source_name] = []
                source_files[source_name].append(file)
        
        # Split the source files into train/val/test
        source_names = list(source_files.keys())
        random.shuffle(source_names)
        
        n_sources = len(source_names)
        n_train = int(train_ratio * n_sources)
        n_val = int(val_ratio * n_sources)
        
        train_sources = source_names[:n_train]
        val_sources = source_names[n_train:n_train+n_val]
        test_sources = source_names[n_train+n_val:]
        
        # Move files to respective directories
        for source in train_sources:
            for file in source_files[source]:
                shutil.copy(file, os.path.join(train_class_dir, os.path.basename(file)))
        
        for source in val_sources:
            for file in source_files[source]:
                shutil.copy(file, os.path.join(val_class_dir, os.path.basename(file)))
        
        for source in test_sources:
            for file in source_files[source]:
                shutil.copy(file, os.path.join(test_class_dir, os.path.basename(file)))
        
        print(f"Class {class_name}: {len(train_sources)} sources in train, {len(val_sources)} in val, {len(test_sources)} in test")
        print(f"  Total images: {sum(len(source_files[s]) for s in train_sources)} in train, "
              f"{sum(len(source_files[s]) for s in val_sources)} in val, "
              f"{sum(len(source_files[s]) for s in test_sources)} in test")

# Example usage
input_dir = "/home/jjpark/Graduation/VisualVroom/direction/side"
file_pattern = "Bike_L*.m4a"
output_dir = "/home/jjpark/Graduation/VisualVroom/direction/train/Bike_L"

# Process audio files and generate images
process_directory(input_dir, file_pattern, output_dir)

# Optional: Create proper train/val/test splits (run this after processing all classes)
# base_dir = "/home/jjpark/Graduation/VisualVroom/direction"
# create_train_val_test_splits(base_dir)