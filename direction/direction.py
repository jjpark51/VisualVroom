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

def save_images(images, output_dir, start_index=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, img in enumerate(images):
        img_path = os.path.join(output_dir, f"sample_{start_index + i}.png")
        img.save(img_path)

def process_directory(input_dir, file_pattern, output_dir):
    file_paths = glob.glob(os.path.join(input_dir, file_pattern))
    start_index = 0
    
    # Determine the starting index to avoid overlapping
    existing_files = glob.glob(os.path.join(output_dir, "sample_*.png"))
    if existing_files:
        existing_indices = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_files]
        start_index = max(existing_indices) + 1
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        # Process audio file
        segments, sample_rate = process_audio_file(file_path)
        
        # Generate images
        images = generate_images(segments, sample_rate)
        
        # Save images
        save_images(images, output_dir, start_index)
        
        # Update start index for the next file
        start_index += len(images)
        
        # Calculate sound direction for each segment
        for i, (top, bottom) in enumerate(segments):
            direction = calculate_sound_direction(top, bottom)
            print(f"Segment {i}: The sound is coming from the {direction}.")

# Example usage
input_dir = "/home/jjpark/Graduation/VisualVroom/direction/side"
file_pattern = "Siren_R*.m4a"
output_dir = "/home/jjpark/Graduation/VisualVroom/direction/train/Siren_R"

process_directory(input_dir, file_pattern, output_dir)