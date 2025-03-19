import os
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
import argparse
import subprocess
import tempfile
import random
import sys
from scipy import signal

def convert_to_wav(input_file):
    """
    Convert an audio file to WAV format if it's in a potentially problematic format (like m4a).
    
    Parameters:
    input_file: Path to the input audio file
    
    Returns:
    temp_file: Path to the temporary WAV file, or original file if no conversion needed
    is_temp: Boolean indicating whether the returned file is a temporary file
    """
    _, ext = os.path.splitext(input_file.lower())
    
    # If the file is already in a well-supported format, return it as is
    if ext in ['.wav', '.mp3', '.ogg', '.flac']:
        return input_file, False
    
    print(f"Converting {input_file} to WAV format for processing...")
    
    # Create a temporary WAV file
    temp_dir = tempfile.gettempdir()
    basename = os.path.basename(input_file)
    temp_name = f"temp_{os.path.splitext(basename)[0]}.wav"
    temp_file = os.path.join(temp_dir, temp_name)
    
    try:
        # Try to load and convert using pydub (most reliable for common formats)
        sound = AudioSegment.from_file(input_file)
        sound.export(temp_file, format="wav")
        return temp_file, True
    except Exception as e:
        print(f"Pydub conversion failed: {e}. Trying ffmpeg...")
        
        try:
            # Try direct ffmpeg conversion as fallback
            subprocess.run([
                "ffmpeg", "-y", "-i", input_file, 
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", 
                temp_file
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return temp_file, True
        except Exception as ffmpeg_error:
            print(f"FFmpeg conversion failed: {ffmpeg_error}. Proceeding with original file.")
            return input_file, False

def detect_speech_boundaries(audio_path, threshold_db=-40, min_silence_duration=0.1):
    """
    Detect the start and end of speech in an audio file based on amplitude thresholds.
    
    Parameters:
    audio_path: Path to the audio file
    threshold_db: Threshold in decibels below which audio is considered silence
    min_silence_duration: Minimum silence duration in seconds
    
    Returns:
    start_time: Time in seconds where speech starts
    end_time: Time in seconds where speech ends
    """
    # Convert file if needed
    processed_path, is_temp = convert_to_wav(audio_path)
    
    try:
        # Load audio file
        y, sr = librosa.load(processed_path, sr=None)
        
        # Convert to decibels
        db = librosa.amplitude_to_db(np.abs(y), ref=np.max)
        
        # Find where audio is above threshold
        mask = db > threshold_db
        
        # Find indices of speech start and end
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return 0, len(y) / sr  # If no speech detected, return the whole file
        
        start_idx = indices[0]
        end_idx = indices[-1]
        
        # Convert indices to time
        start_time = start_idx / sr
        end_time = end_idx / sr
        
        return start_time, end_time
    finally:
        # Clean up temporary file if one was created
        if is_temp and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except:
                pass

def generate_output_path(input_path, output_dir=None):
    """
    Generate output path by adding '-Trimmed' suffix to the filename.
    
    Parameters:
    input_path: Original file path
    output_dir: Optional output directory
    
    Returns:
    output_path: Path for the trimmed file
    """
    # Get the directory, filename, and extension
    dir_path, filename = os.path.split(input_path)
    basename, ext = os.path.splitext(filename)
    
    # If output directory is specified, use it; otherwise use the same directory as input
    output_directory = output_dir if output_dir else dir_path
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Generate new filename with '-Trimmed' suffix
    new_filename = f"{basename}-Trimmed{ext}"
    output_path = os.path.join(output_directory, new_filename)
    
    return output_path

def apply_audio_effects(y, sr, audio_type, effects):
    """
    Apply various audio effects to make AI and human recordings sound more similar.
    
    Parameters:
    y: Audio time series
    sr: Sample rate
    audio_type: 'human' or 'ai' - the type of audio being processed
    effects: Dictionary of effect settings
    
    Returns:
    y_processed: Processed audio time series
    """
    y_processed = np.copy(y)
    
    # 1. Add background noise if enabled
    if effects['add_noise']:
        noise_level = effects['noise_level'] * np.sqrt(np.mean(y_processed**2))
        noise = np.random.normal(0, noise_level, len(y_processed))
        y_processed = y_processed + noise
    
    # 2. Apply EQ if enabled
    if effects['apply_eq']:
        if audio_type == 'ai':
            if random.random() < 0.7:  # Apply to most but not all AI files
                # Make AI recordings sound slightly less perfect by reducing high frequencies
                b, a = signal.butter(4, 0.8, 'lowpass')
                y_processed = signal.filtfilt(b, a, y_processed)
        else:  # human
            if random.random() < 0.5:  # Apply to some human files
                # Sometimes clean up human recordings slightly
                y_processed = librosa.effects.preemphasis(y_processed, coef=0.1)
    
    # 3. Add room reverb if enabled
    if effects['add_reverb']:
        if audio_type == 'ai' and random.random() < 0.8:
            # Add subtle reverb to AI recordings to make them sound more natural
            reverb_amount = random.uniform(0.05, 0.15)
            # Simple impulse response for a small room
            impulse_length = int(sr * 0.2)  # 200ms reverb tail
            impulse_response = np.exp(-np.linspace(0, 8, impulse_length))
            reverb = signal.convolve(y_processed, impulse_response, mode='full')[:len(y_processed)]
            y_processed = (1 - reverb_amount) * y_processed + reverb_amount * reverb
    
    # 4. Add subtle artifacts if enabled
    if effects['add_artifacts']:
        if audio_type == 'human' and random.random() < 0.6:
            # Add very subtle digital artifacts to human recordings
            # to match some artifacts in AI recordings
            bit_reduction = 14 + random.randint(0, 2)  # Very mild bit reduction (14-16 bits)
            y_processed = np.round(y_processed * (2 ** bit_reduction)) / (2 ** bit_reduction)
    
    # 5. Add subtle pitch variation if enabled
    if effects['pitch_variation']:
        if (audio_type == 'ai' and random.random() < 0.7) or (audio_type == 'human' and random.random() < 0.3):
            # More pitch variation for AI than human
            cents = random.uniform(-20, 20)  # Small variation
            y_processed = librosa.effects.pitch_shift(y_processed, sr=sr, n_steps=cents/100)
    
    return y_processed

def trim_audio(input_file, output_file=None, start_time=None, end_time=None, padding=0.1, 
              normalize_volume=False, target_dBFS=-20, audio_type=None, effects=None):
    """
    Trim an audio file to the specified start and end times with padding,
    and optionally apply audio effects to make AI and human sound similar.
    
    Parameters:
    input_file: Path to the input audio file
    output_file: Path to save the trimmed audio (if None, it's auto-generated)
    start_time: Start time in seconds (if None, it's detected)
    end_time: End time in seconds (if None, it's detected)
    padding: Additional padding in seconds to include before and after detected speech
    normalize_volume: Whether to normalize volume
    target_dBFS: Target volume level in dBFS
    audio_type: 'human' or 'ai' - the type of audio being processed
    effects: Dictionary of effect settings
    
    Returns:
    output_path: Path where the trimmed file was saved
    """
    # If no output file specified, generate one
    if output_file is None:
        output_file = generate_output_path(input_file)
    
    # If start and end times are not provided, detect them
    if start_time is None or end_time is None:
        start_time, end_time = detect_speech_boundaries(input_file)
    
    # Convert file if needed - use same format for output if possible
    processed_path, is_temp = convert_to_wav(input_file)
    
    try:
        # First load with pydub for trimming and volume adjustment
        sound = AudioSegment.from_file(processed_path)
        
        # Add padding (but ensure we don't go below 0 or beyond the file duration)
        start_ms = max(0, int((start_time - padding) * 1000))
        end_ms = min(len(sound), int((end_time + padding) * 1000))
        
        # Extract the portion we want
        trimmed_sound = sound[start_ms:end_ms]
        
        # Normalize volume if requested
        if normalize_volume:
            change_in_dBFS = target_dBFS - trimmed_sound.dBFS
            trimmed_sound = trimmed_sound.apply_gain(change_in_dBFS)
        
        # Export to temporary file for further processing with librosa if effects are requested
        if effects is not None and audio_type is not None:
            temp_trimmed = os.path.join(tempfile.gettempdir(), "temp_trimmed.wav")
            trimmed_sound.export(temp_trimmed, format="wav")
            
            # Load with librosa for advanced effects
            y, sr = librosa.load(temp_trimmed, sr=None)
            
            # Apply audio effects
            y_processed = apply_audio_effects(y, sr, audio_type, effects)
            
            # Save processed audio
            sf.write(temp_trimmed, y_processed, sr)
            
            # Load back into pydub for final export
            trimmed_sound = AudioSegment.from_file(temp_trimmed)
            
            # Clean up temporary file
            os.remove(temp_trimmed)
        
        # Determine output format based on extension
        output_format = os.path.splitext(output_file)[1][1:]
        
        # If output format is not supported, default to wav
        if output_format.lower() in ['m4a']:
            print(f"Warning: Output format {output_format} may not be directly supported. Using FFMPEG if available.")
            
            # First export as WAV
            temp_wav = os.path.join(tempfile.gettempdir(), "temp_export.wav")
            trimmed_sound.export(temp_wav, format="wav")
            
            # Then convert to the desired format using ffmpeg
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", temp_wav, output_file
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                os.remove(temp_wav)
            except Exception as e:
                print(f"Warning: FFMPEG conversion failed: {e}. Exporting as WAV instead.")
                # Fall back to WAV output with modified filename
                output_file = f"{os.path.splitext(output_file)[0]}.wav"
                trimmed_sound.export(output_file, format="wav")
        else:
            # Export the trimmed audio in the original format
            trimmed_sound.export(output_file, format=output_format)
        
        return output_file
    finally:
        # Clean up temporary file if one was created
        if is_temp and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except:
                pass

def process_directory(input_dir, output_dir=None, threshold_db=-40, padding=0.1, 
                     normalize_volume=False, target_dBFS=-20, audio_type=None, effects=None):
    """
    Process all audio files in a directory.
    
    Parameters:
    input_dir: Directory containing audio files
    output_dir: Directory to save trimmed audio files (if None, files are saved in the input directory)
    threshold_db: Threshold for speech detection
    padding: Additional padding in seconds
    normalize_volume: Whether to normalize volume
    target_dBFS: Target volume level in dBFS
    audio_type: 'human' or 'ai' - the type of audio being processed
    effects: Dictionary of effect settings
    """
    # Create output directory if specified and it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
    
    for audio_file in audio_files:
        input_path = os.path.join(input_dir, audio_file)
        
        # Generate output path (either in specified output directory or in the input directory with modified name)
        if output_dir:
            basename, ext = os.path.splitext(audio_file)
            new_filename = f"{basename}-Trimmed{ext}"
            output_path = os.path.join(output_dir, new_filename)
        else:
            output_path = generate_output_path(input_path)
        
        print(f"Processing {audio_file}...")
        
        try:
            # Detect speech boundaries
            start_time, end_time = detect_speech_boundaries(input_path, threshold_db)
            
            # Trim audio and apply effects if requested
            trim_audio(input_path, output_path, start_time, end_time, padding, 
                      normalize_volume, target_dBFS, audio_type, effects)
            
            print(f"Successfully trimmed {audio_file} to {os.path.basename(output_path)}")
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

def match_volume(files, target_dBFS=-20, output_dir=None):
    """
    Match the volume of multiple audio files and save them to an output directory.
    
    Parameters:
    files: List of file paths
    target_dBFS: Target volume level in dBFS
    output_dir: Directory to save volume-matched files
    """
    # Ensure output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for file in files:
        # Convert file if needed
        processed_path, is_temp = convert_to_wav(file)
        
        try:
            sound = AudioSegment.from_file(processed_path)
            change_in_dBFS = target_dBFS - sound.dBFS
            normalized = sound.apply_gain(change_in_dBFS)
            
            # Generate output filename (always WAV format)
            base_name = os.path.basename(file)
            base_without_ext = os.path.splitext(base_name)[0]
            output_file = os.path.join(output_dir, f"{base_without_ext}-Normalized.wav")
            
            # Export as WAV
            normalized.export(output_file, format="wav")
            print(f"Normalized volume for {file} to {target_dBFS} dBFS, saved to {output_file}")
            
        except Exception as e:
            print(f"Error normalizing {file}: {e}")
        finally:
            # Clean up temporary file if one was created
            if is_temp and os.path.exists(processed_path):
                try:
                    os.remove(processed_path)
                except:
                    pass

def match_audio_characteristics(human_folder, ai_folder, output_dir=None, options=None):
    """
    Process both human and AI audio to make them sound more similar.
    
    Parameters:
    human_folder: Folder containing human recordings
    ai_folder: Folder containing AI recordings
    output_dir: Output directory for processed files
    options: Dictionary of processing options
    """
    # Set default options if none provided
    if options is None:
        options = {
            'normalize': True,
            'volume': -20,
            'threshold': -40,
            'padding': 0.1,
            'add_noise': False,
            'noise_level': 0.03,
            'apply_eq': False,
            'add_reverb': False,
            'add_artifacts': False,
            'pitch_variation': False
        }
    
    # Create output folders
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(human_folder), "processed")
    
    human_output = os.path.join(output_dir, "Human-processed")
    ai_output = os.path.join(output_dir, "AI-processed")
    
    print(f"Processing human files from {human_folder} to {human_output}...")
    # Process human files
    effects = {
        'add_noise': options['add_noise'],
        'noise_level': options['noise_level'],
        'apply_eq': options['apply_eq'],
        'add_reverb': options['add_reverb'],
        'add_artifacts': options['add_artifacts'],
        'pitch_variation': options['pitch_variation']
    }
    
    process_directory(
        human_folder, 
        human_output, 
        options['threshold'], 
        options['padding'], 
        options['normalize'], 
        options['volume'], 
        'human', 
        effects
    )
    
    print(f"Processing AI files from {ai_folder} to {ai_output}...")
    # Process AI files
    process_directory(
        ai_folder, 
        ai_output, 
        options['threshold'], 
        options['padding'], 
        options['normalize'], 
        options['volume'], 
        'ai', 
        effects
    )
    
    print(f"Audio processing complete. Processed files saved to {output_dir}")

# Add this function near the bottom of your script, just before the if __name__ == "__main__": section:

def interactive_mode():
    """Run the script in interactive mode, guiding the user through options via the terminal."""
    print("\n" + "="*60)
    print("AUDIO PROCESSING TOOL - INTERACTIVE MODE")
    print("="*60)
    
    # Ask what the user wants to do
    print("\nWhat would you like to do?")
    print("1. Trim audio file(s) and remove silence")
    print("2. Match volume levels across audio files")
    print("3. Process human and AI recordings to sound more similar")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        # TRIM AUDIO FILES
        print("\n" + "-"*60)
        print("TRIM AUDIO FILES")
        print("-"*60)
        
        # Get input files
        print("\nEnter the path(s) to your audio file(s) or directory.")
        print("For multiple files, separate paths with commas.")
        input_paths_str = input("Path(s): ").strip()
        input_paths = [path.strip() for path in input_paths_str.split(",")]
        
        # Check if valid paths
        valid_paths = []
        for path in input_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                print(f"Warning: Path not found: {path}")
        
        if not valid_paths:
            print("No valid paths provided. Exiting.")
            return
        
        # Output directory
        output_dir = input("\nOutput directory (leave blank for same directory as input): ").strip()
        if output_dir and not os.path.exists(output_dir):
            create_dir = input(f"Directory {output_dir} doesn't exist. Create it? (y/n): ").lower()
            if create_dir == 'y':
                os.makedirs(output_dir)
            else:
                output_dir = ""
        
        # Get threshold
        threshold_str = input("\nSilence threshold in dB (leave blank for default -40): ").strip()
        threshold = -40
        if threshold_str:
            try:
                threshold = float(threshold_str)
            except ValueError:
                print("Invalid threshold. Using default -40 dB.")
        
        # Get padding
        padding_str = input("\nPadding in seconds (leave blank for default 0.1): ").strip()
        padding = 0.1
        if padding_str:
            try:
                padding = float(padding_str)
            except ValueError:
                print("Invalid padding. Using default 0.1 seconds.")
        
        # Normalize volume
        normalize = input("\nNormalize volume? (y/n, default: n): ").lower().startswith('y')
        
        # Get target volume if normalizing
        volume = -20
        if normalize:
            volume_str = input("Target volume in dBFS (leave blank for default -20): ").strip()
            if volume_str:
                try:
                    volume = float(volume_str)
                except ValueError:
                    print("Invalid volume. Using default -20 dBFS.")
        
        # Output format
        print("\nSelect output format:")
        print("1. Same as input (default)")
        print("2. WAV")
        print("3. MP3")
        print("4. OGG")
        print("5. FLAC")
        format_choice = input("Enter choice (1-5): ").strip()
        
        format_map = {
            "2": "wav",
            "3": "mp3", 
            "4": "ogg",
            "5": "flac"
        }
        output_format = format_map.get(format_choice, "original")
        
        # Process files
        print("\nProcessing files...")
        for input_path in valid_paths:
            if os.path.isdir(input_path):
                print(f"Processing directory: {input_path}")
                process_directory(input_path, output_dir, threshold, padding, normalize, volume)
            else:
                try:
                    print(f"Processing file: {input_path}")
                    # Detect speech boundaries
                    start_time, end_time = detect_speech_boundaries(input_path, threshold)
                    
                    # Determine output file path
                    output_file = None
                    if output_format != 'original':
                        # Generate path but change extension
                        if output_dir:
                            base_name = os.path.basename(input_path)
                            base_without_ext = os.path.splitext(base_name)[0]
                            output_file = os.path.join(output_dir, f"{base_without_ext}-Trimmed.{output_format}")
                        else:
                            base_output = generate_output_path(input_path)
                            base_without_ext = os.path.splitext(base_output)[0]
                            output_file = f"{base_without_ext}.{output_format}"
                    elif output_dir:
                        base_name = os.path.basename(input_path)
                        output_file = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}-Trimmed{os.path.splitext(base_name)[1]}")
                    
                    # Trim audio
                    result_file = trim_audio(input_path, output_file, start_time, end_time, padding, normalize, volume)
                    print(f"Successfully trimmed to: {result_file}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
        
        print("\nTrimming complete!")
    
    elif choice == "2":
        # MATCH VOLUME
        print("\n" + "-"*60)
        print("MATCH VOLUME LEVELS")
        print("-"*60)
        
        # Get input files
        print("\nEnter the path(s) to your audio file(s).")
        print("For multiple files, separate paths with commas.")
        input_paths_str = input("Path(s): ").strip()
        input_paths = [path.strip() for path in input_paths_str.split(",")]
        
        # Check if valid paths
        valid_paths = []
        for path in input_paths:
            if os.path.exists(path) and os.path.isfile(path):
                valid_paths.append(path)
            else:
                print(f"Warning: Invalid file path: {path}")
        
        if not valid_paths:
            print("No valid files provided. Exiting.")
            return
        
        # Require output directory
        output_dir = input("\nOutput directory (required): ").strip()
        if not output_dir:
            print("Output directory is required. Exiting.")
            return
            
        if not os.path.exists(output_dir):
            create_dir = input(f"Directory {output_dir} doesn't exist. Create it? (y/n): ").lower()
            if create_dir == 'y':
                os.makedirs(output_dir)
            else:
                print("Output directory required. Exiting.")
                return
        
        # Get target volume
        volume_str = input("\nTarget volume in dBFS (leave blank for default -20): ").strip()
        volume = -20
        if volume_str:
            try:
                volume = float(volume_str)
            except ValueError:
                print("Invalid volume. Using default -20 dBFS.")
        
        # Process files
        print("\nMatching volume levels...")
        match_volume(valid_paths, volume, output_dir)
        print("\nVolume matching complete! All files saved as WAV format.")
    
    elif choice == "3":
        # MATCH AUDIO CHARACTERISTICS
        print("\n" + "-"*60)
        print("PROCESS HUMAN AND AI RECORDINGS")
        print("-"*60)
        
        # Get human folder
        human_folder = input("\nEnter the path to the folder with HUMAN recordings: ").strip()
        if not os.path.isdir(human_folder):
            print(f"Error: {human_folder} is not a valid directory. Exiting.")
            return
        
        # Get AI folder
        ai_folder = input("\nEnter the path to the folder with AI recordings: ").strip()
        if not os.path.isdir(ai_folder):
            print(f"Error: {ai_folder} is not a valid directory. Exiting.")
            return
        
        # Output directory
        output_dir = input("\nOutput directory (leave blank for default 'processed' subfolder): ").strip()
        
        # Get processing options
        options = {}
        
        # Normalize volume (on by default)
        normalize = input("\nNormalize volume? (y/n, default: y): ").lower()
        options['normalize'] = False if normalize == 'n' else True
        
        # Get target volume if normalizing
        options['volume'] = -20
        if options['normalize']:
            volume_str = input("Target volume in dBFS (leave blank for default -20): ").strip()
            if volume_str:
                try:
                    options['volume'] = float(volume_str)
                except ValueError:
                    print("Invalid volume. Using default -20 dBFS.")
        
        # Silence detection options
        threshold_str = input("\nSilence threshold in dB (leave blank for default -40): ").strip()
        options['threshold'] = -40
        if threshold_str:
            try:
                options['threshold'] = float(threshold_str)
            except ValueError:
                print("Invalid threshold. Using default -40 dB.")
        
        padding_str = input("\nPadding in seconds (leave blank for default 0.1): ").strip()
        options['padding'] = 0.1
        if padding_str:
            try:
                options['padding'] = float(padding_str)
            except ValueError:
                print("Invalid padding. Using default 0.1 seconds.")
        
        print("\nEnhancement options (all default to no):")
        
        # Background noise
        options['add_noise'] = input("Add subtle background noise? (y/n): ").lower().startswith('y')
        if options['add_noise']:
            noise_str = input("Noise level (0.01-0.1, leave blank for default 0.03): ").strip()
            options['noise_level'] = 0.03
            if noise_str:
                try:
                    options['noise_level'] = float(noise_str)
                except ValueError:
                    print("Invalid noise level. Using default 0.03.")
        
        # EQ
        options['apply_eq'] = input("Apply EQ to match frequency response? (y/n): ").lower().startswith('y')
        
        # Reverb
        options['add_reverb'] = input("Add subtle room reverb to AI recordings? (y/n): ").lower().startswith('y')
        
        # Artifacts
        options['add_artifacts'] = input("Add subtle digital artifacts to humanize AI recordings? (y/n): ").lower().startswith('y')
        
        # Pitch variation
        options['pitch_variation'] = input("Add subtle pitch variations? (y/n): ").lower().startswith('y')
        
        # Process the files
        print("\nProcessing files...")
        match_audio_characteristics(human_folder, ai_folder, output_dir, options)
        print("\nProcessing complete!")
    
    elif choice == "4":
        print("\nExiting...")
        return
    else:
        print("\nInvalid choice. Please try again.")
        interactive_mode()

# Modify the if __name__ == "__main__" block to include the interactive option:

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # Run in interactive mode
        interactive_mode()
    else:
        # Regular argument parsing
        parser = argparse.ArgumentParser(description='Trim audio files to remove silence before and after speech.')
        
        # Add interactive mode option
        parser.add_argument('--interactive', action='store_true', 
                          help='Run in interactive mode with text-based prompts')
        
        # Create subparsers for different commands
        subparsers = parser.add_subparsers(dest='command', help='Command to execute')
        
        # Standard trimming command
        trim_parser = subparsers.add_parser('trim', help='Trim audio files')
        trim_parser.add_argument('input', nargs='+', help='Input audio file(s) or directory')
        trim_parser.add_argument('--outdir', '-o', help='Output directory (optional). If not provided, files are saved in the same directory with -Trimmed suffix')
        trim_parser.add_argument('--threshold', '-t', type=float, default=-40, 
                                help='Threshold in dB below which audio is considered silence (default: -40)')
        trim_parser.add_argument('--padding', '-p', type=float, default=0.1, 
                                help='Padding in seconds to add before and after detected speech (default: 0.1)')
        trim_parser.add_argument('--normalize', '-n', action='store_true',
                                help='Normalize volume of the output files')
        trim_parser.add_argument('--volume', '-v', type=float, default=-20,
                                help='Target volume level in dBFS (default: -20)')
        trim_parser.add_argument('--output-format', choices=['wav', 'mp3', 'ogg', 'flac', 'm4a', 'original'],
                                default='original', help='Force a specific output format (default: same as input)')
        
        # Volume matching command
        vol_parser = subparsers.add_parser('match-volume', help='Match volume levels across files')
        vol_parser.add_argument('input', nargs='+', help='Input audio file(s)')
        vol_parser.add_argument('--outdir', '-o', required=True, help='Output directory for volume-matched files')
        vol_parser.add_argument('--volume', '-v', type=float, default=-20,
                            help='Target volume level in dBFS (default: -20)')
        
        # Match audio characteristics command
        match_parser = subparsers.add_parser('match-characteristics', 
                                            help='Process human and AI recordings to sound more similar')
        match_parser.add_argument('--human', required=True, help='Folder containing human recordings')
        match_parser.add_argument('--ai', required=True, help='Folder containing AI recordings')
        match_parser.add_argument('--outdir', '-o', help='Output directory for processed files')
        match_parser.add_argument('--threshold', '-t', type=float, default=-40, 
                                 help='Threshold in dB below which audio is considered silence (default: -40)')
        match_parser.add_argument('--padding', '-p', type=float, default=0.1, 
                                 help='Padding in seconds to add before and after detected speech (default: 0.1)')
        match_parser.add_argument('--normalize', '-n', action='store_true', default=True,
                                 help='Normalize volume of the output files')
        match_parser.add_argument('--volume', '-v', type=float, default=-20,
                                 help='Target volume level in dBFS (default: -20)')
        match_parser.add_argument('--add-noise', action='store_true', default=False,
                                 help='Add subtle background noise')
        match_parser.add_argument('--noise-level', type=float, default=0.03,
                                 help='Background noise level (0.01-0.1 recommended)')
        match_parser.add_argument('--apply-eq', action='store_true', default=False,
                                 help='Apply equalization to match spectral qualities')
        match_parser.add_argument('--add-reverb', action='store_true', default=False,
                                 help='Add room reverb effects')
        match_parser.add_argument('--add-artifacts', action='store_true', default=False,
                                 help='Add subtle compression artifacts')
        match_parser.add_argument('--pitch-variation', action='store_true', default=False,
                                 help='Add subtle pitch variations')
        
        args = parser.parse_args()
        
        # Check if interactive mode was requested
        if hasattr(args, 'interactive') and args.interactive:
            interactive_mode()
        else:
            # For backwards compatibility, if no command is specified, assume 'trim'
            if args.command is None:
                if hasattr(args, 'input'):  # Old-style command line
                    args.command = 'trim'
                else:
                    parser.print_help()
                    exit(1)
            
            if args.command == 'trim':
                # [existing trim command code]
                # Process each input
                for input_path in args.input:
                    if os.path.isdir(input_path):
                        # Process entire directory
                        process_directory(input_path, args.outdir, args.threshold, args.padding, 
                                        args.normalize, args.volume)
                    else:
                        # Process single file
                        try:
                            # Detect speech boundaries
                            start_time, end_time = detect_speech_boundaries(input_path, args.threshold)
                            
                            # Determine output file path
                            output_file = None
                            if args.output_format != 'original':
                                # Generate path but change extension
                                base_output = generate_output_path(input_path)
                                base_without_ext = os.path.splitext(base_output)[0]
                                output_file = f"{base_without_ext}.{args.output_format}"
                            
                            # Trim audio
                            output_file = trim_audio(input_path, output_file, start_time, end_time, 
                                                args.padding, args.normalize, args.volume)
                            
                            print(f"Successfully trimmed {input_path} to {output_file}")
                        except Exception as e:
                            print(f"Error processing {input_path}: {e}")
            
            elif args.command == 'match-volume':
                match_volume(args.input, args.volume, args.outdir)
            
            elif args.command == 'match-characteristics':
                options = {
                    'normalize': args.normalize,
                    'volume': args.volume,
                    'threshold': args.threshold,
                    'padding': args.padding,
                    'add_noise': args.add_noise,
                    'noise_level': args.noise_level,
                    'apply_eq': args.apply_eq,
                    'add_reverb': args.add_reverb,
                    'add_artifacts': args.add_artifacts,
                    'pitch_variation': args.pitch_variation
                }
                
                match_audio_characteristics(args.human, args.ai, args.outdir, options)