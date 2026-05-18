import os
import shutil
import pandas as pd
import subprocess
import argparse
import ast
import soundfile as sf
import librosa

# Configuration matching infer.sh
DATASET = "MELD_clips"

# CSV maps for different datasets
MELD_TRANSCRIPT_CSV = "/data2/ruixin/datasets/MELD_clips/metadata.csv"
V2C_TRANSCRIPT_CSV = "/data2/ruixin/datasets/v2c_clips/metadata.csv"
CHEM_TRANSCRIPT_CSV = "/data2/ruixin/datasets/chem_clips/metadata.csv"

# Base path for datasets (from infer.sh)
DATA_ROOT = "/data2/ruixin/datasets"

def parse_list_field(value):
    """Parses a string representation of a list into a python list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    s = str(value).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        return [s]

def resample_and_save(src_path, dst_path, target_sr=16000):
    """
    Reads audio from src_path, resamples to target_sr, and saves to dst_path.
    Returns True if successful, False otherwise.
    """
    try:
        # Load with librosa, which handles resampling and mono checking automatically
        # mono=True because MELD clips are mono.
        y, sr = librosa.load(src_path, sr=target_sr, mono=True)
        sf.write(dst_path, y, target_sr)
        return True
    except Exception as e:
        print(f"Error resampling {src_path}: {e}")
        return False

def calculate_energy_db(audio_path):
    """
    Calculate the energy of an audio file in dB.
    Returns the energy in dB, or None if error occurs.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if len(y) == 0:
            return float('-inf')
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]
        # Convert RMS to dB, then take mean
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)
        mean_rms_db = rms_db.mean()
        return mean_rms_db
    except Exception as e:
        print(f"Error calculating energy for {audio_path}: {e}")
        return None

def convert_to_mono(src_path, dst_path):
    """
    Convert multi-channel audio to mono and save.
    Returns True if successful, False otherwise.
    """
    try:
        y, sr = librosa.load(src_path, sr=None, mono=True)
        sf.write(dst_path, y, sr)
        return True
    except Exception as e:
        print(f"Error converting to mono {src_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run inference and post-processing for audio separation.")
    parser.add_argument("--dataset", type=str, default="MELD_clips", choices=["MELD_clips", "v2c_clips", "chem_clips"], help="Dataset name")
    args = parser.parse_args()

    dataset = args.dataset
    
    # Get CSV path
    if dataset == "MELD_clips":
        csv_path = MELD_TRANSCRIPT_CSV
    elif dataset == "v2c_clips":
        csv_path = V2C_TRANSCRIPT_CSV
    elif dataset == "chem_clips":
        csv_path = CHEM_TRANSCRIPT_CSV
    else:
        print(f"Error: Unsupported dataset '{dataset}'")
        return
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Define paths
    if dataset == "MELD_clips":
        # MELD: unified folders, no split subdirectories
        input_folder = os.path.join(DATA_ROOT, dataset, "audios", "ost")
        store_dir = os.path.join(DATA_ROOT, dataset, "audios", "vocals")
        ins_dir = os.path.join(DATA_ROOT, dataset, "audios", "ins")
    elif dataset == "v2c_clips":
        input_folder = os.path.join(DATA_ROOT, dataset, "audios", "ost")
        store_dir = os.path.join(DATA_ROOT, dataset, "audios", "vocals")
        ins_dir = os.path.join(DATA_ROOT, dataset, "audios", "ins")
    else:  # chem_clips
        input_folder = os.path.join(DATA_ROOT, dataset, "audios", "ost")
        store_dir = os.path.join(DATA_ROOT, dataset, "audios", "vocals")
        ins_dir = os.path.join(DATA_ROOT, dataset, "audios", "ins")
    
    model_path = "/data2/ruixin/downloads/melbandroformer/MelBandRoformer.ckpt"
    config_path = "configs/config_vocals_mel_band_roformer.yaml"

    print(f"Processing dataset: {dataset}")
    print(f"Input folder: {input_folder}")
    print(f"Store directory: {store_dir}")
    print(f"Instrumental directory: {ins_dir}")

    # 0. Pre-processing: Convert multi-channel audio to mono
    print("\nStarting pre-processing (converting to mono)...")
    mono_converted = 0
    if os.path.exists(input_folder):
        for audio_file in os.listdir(input_folder):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(input_folder, audio_file)
                try:
                    # Check if audio is already mono
                    info = sf.info(audio_path)
                    if info.channels > 1:
                        # Convert to mono in place
                        temp_path = audio_path + ".tmp.wav"
                        if convert_to_mono(audio_path, temp_path):
                            shutil.move(temp_path, audio_path)
                            mono_converted += 1
                except Exception as e:
                    print(f"Warning: Could not process {audio_file}: {e}")
    print(f"Converted {mono_converted} multi-channel audio files to mono.")

    # 1. Execute Inference (Mel-RoFormer separation)
    cmd = [
        "python", "inference.py",
        "--config_path", config_path,
        "--model_path", model_path,
        "--input_folder", input_folder,
        "--store_dir", store_dir
    ]

    print("\nStarting inference...")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")
        return

    # 2. Post-processing: Move instrumentals and create text files
    print("\nStarting post-processing (moving files and creating text files)...")
    
    if not os.path.exists(ins_dir):
        os.makedirs(ins_dir, exist_ok=True)

    # Read transcripts
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return

    # Check if CSV has required columns
    if dataset == "MELD_clips" and 'Split' not in df.columns:
        print(f"Error: metadata.csv missing 'Split' column")
        return
    if dataset == "MELD_clips" and 'Sample_ID' not in df.columns:
        print(f"Error: metadata.csv missing 'Sample_ID' column")
        return
    if dataset == "chem_clips" and 'Sample_ID' not in df.columns:
        print(f"Error: metadata.csv missing 'Sample_ID' column")
        return

    moved_count = 0
    txt_count = 0
    filtered_count = 0
    # 只过滤极端静音的样本（-60 dB 以下通常表示几乎没有信号）
    # 参考：正常语音通常在 -30 到 -50 dB，背景音乐在 -20 到 -60 dB
    energy_threshold_db = -50.0  # Energy threshold in dB

    if dataset == "MELD_clips":
        # Process all samples in the dataset
        # Iterate through all rows
        for idx, row in df.iterrows():
            sample_id = row.get('Sample_ID')
            split = row.get('Split', 'unknown')
            if pd.isna(sample_id):
                continue
            
            # Use split from metadata to construct base_name
            base_name = f"{split}_sample_{int(sample_id)}"
            
            # Inferred filenames
            # inference.py appends _{instrument}.wav
            vocals_filename = f"{base_name}_vocals.wav"
            instrumental_filename = f"{base_name}_instrumental.wav"

            vocals_path = os.path.join(store_dir, vocals_filename)
            instrumental_src = os.path.join(store_dir, instrumental_filename)
            instrumental_dst = os.path.join(ins_dir, instrumental_filename)

            # Parse Utterances list and join them
            utterances_raw = row.get('Utterances')
            utterances_list = parse_list_field(utterances_raw)
            utterance_text = " ".join([str(u) for u in utterances_list])

            # Check if both files exist before processing
            if not os.path.exists(vocals_path) or not os.path.exists(instrumental_src):
                continue
            
            # Energy-based filtering: calculate energy for vocals
            vocals_energy = calculate_energy_db(vocals_path)
            
            # Filter out samples with low vocal energy (silent speech)
            if vocals_energy is None:
                print(f"Warning: Could not calculate energy for {base_name}, skipping.")
                filtered_count += 1
                continue
            
            if vocals_energy < energy_threshold_db:
                print(f"Filtered out {base_name}: vocals_energy={vocals_energy:.2f}dB")
                # Remove both files
                if os.path.exists(vocals_path):
                    os.remove(vocals_path)
                if os.path.exists(instrumental_src):
                    os.remove(instrumental_src)
                filtered_count += 1
                continue
            
            # Move and resample instrumental file
            if os.path.exists(instrumental_src):
                # Resample and save to destination
                if resample_and_save(instrumental_src, instrumental_dst, target_sr=16000):
                    # Remove original if successful
                    os.remove(instrumental_src)
                    moved_count += 1
                else:
                    # Fallback: Just move if resampling fails
                    print(f"Warning: Resampling failed for {instrumental_src}, moving directly.")
                    shutil.move(instrumental_src, instrumental_dst)
                    moved_count += 1

            # Resample vocals file in place
            if os.path.exists(vocals_path):
                # We use a temp file or overwrite? librosa load then write should be fine if memory fits
                # But safer to write to temp then rename
                temp_vocals = vocals_path + ".tmp.wav"
                if resample_and_save(vocals_path, temp_vocals, target_sr=16000):
                    shutil.move(temp_vocals, vocals_path)
                
                txt_path = os.path.join(store_dir, f"{base_name}_vocals.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(utterance_text)
                txt_count += 1
    
    elif dataset == "chem_clips":
        # chem_clips: uses Video_ID and Cut_Number to build filenames
        for idx, row in df.iterrows():
            sample_id = row.get('Sample_ID')
            if pd.isna(sample_id):
                continue
            
            # Build filename from Video_ID and Cut_Number
            video_id = row.get('Video_ID', '').strip()
            cut_number = row.get('Cut_Number', '')
            if not video_id or pd.isna(cut_number):
                continue
            
            # Construct base_name: {Video_ID}_cut{Cut_Number}
            base_name = f"{video_id}_cut{int(cut_number)}"
            
            vocals_filename = f"{base_name}_vocals.wav"
            instrumental_filename = f"{base_name}_instrumental.wav"

            vocals_path = os.path.join(store_dir, vocals_filename)
            instrumental_src = os.path.join(store_dir, instrumental_filename)
            instrumental_dst = os.path.join(ins_dir, instrumental_filename)

            # Get utterances - chem uses 'Transcription' field
            utterance_text = ""
            if 'Transcription' in row and pd.notna(row.get('Transcription')):
                utterance_text = str(row.get('Transcription', '')).strip()
            elif 'Utterance' in row and pd.notna(row.get('Utterance')):
                utterance_text = str(row.get('Utterance', '')).strip()
            elif 'Utterances' in row:
                utterances_raw = row.get('Utterances')
                utterances_list = parse_list_field(utterances_raw)
                utterance_text = " ".join([str(u) for u in utterances_list])

            # Check if both files exist before processing
            if not os.path.exists(vocals_path) or not os.path.exists(instrumental_src):
                continue
            
            # Energy-based filtering: calculate energy for vocals
            vocals_energy = calculate_energy_db(vocals_path)
            
            # Filter out samples with low vocal energy (silent speech)
            if vocals_energy is None:
                print(f"Warning: Could not calculate energy for {base_name}, skipping.")
                filtered_count += 1
                continue
            
            if vocals_energy < energy_threshold_db:
                print(f"Filtered out {base_name}: vocals_energy={vocals_energy:.2f}dB")
                # Remove both files
                if os.path.exists(vocals_path):
                    os.remove(vocals_path)
                if os.path.exists(instrumental_src):
                    os.remove(instrumental_src)
                filtered_count += 1
                continue
            
            # Move and resample instrumental file
            if os.path.exists(instrumental_src):
                if resample_and_save(instrumental_src, instrumental_dst, target_sr=16000):
                    os.remove(instrumental_src)
                    moved_count += 1
                else:
                    print(f"Warning: Resampling failed for {instrumental_src}, moving directly.")
                    shutil.move(instrumental_src, instrumental_dst)
                    moved_count += 1

            # Resample vocals file in place
            if os.path.exists(vocals_path):
                temp_vocals = vocals_path + ".tmp.wav"
                if resample_and_save(vocals_path, temp_vocals, target_sr=16000):
                    shutil.move(temp_vocals, vocals_path)
                
                txt_path = os.path.join(store_dir, f"{base_name}_vocals.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(utterance_text)
                txt_count += 1
    
    elif dataset == "v2c_clips":
        # v2c: all files are in the same folder, no need to filter by movie
        # Iterate through each row (each row is one audio clip)
        for idx, row in df.iterrows():
            # Get Audio_Filename column value (e.g., Brave_sample_1_66-67.wav)
            audio_filename = row.get('Audio_Filename', '').strip()
            if not audio_filename:
                continue
            
            # Remove .wav extension to get base_name
            base_name = os.path.splitext(audio_filename)[0]
            
            vocals_filename = f"{base_name}_vocals.wav"
            instrumental_filename = f"{base_name}_instrumental.wav"

            vocals_path = os.path.join(store_dir, vocals_filename)
            instrumental_src = os.path.join(store_dir, instrumental_filename)
            instrumental_dst = os.path.join(ins_dir, instrumental_filename)

            # Get utterances from Utterances column
            utterances_raw = row.get('Utterances')
            utterances_list = parse_list_field(utterances_raw)
            utterance_text = " ".join([str(u) for u in utterances_list])

            # Check if both files exist before processing
            if not os.path.exists(vocals_path) or not os.path.exists(instrumental_src):
                continue
            
            # Energy-based filtering: calculate energy for vocals
            vocals_energy = calculate_energy_db(vocals_path)
            
            # Filter out samples with low vocal energy (silent speech)
            if vocals_energy is None:
                print(f"Warning: Could not calculate energy for {base_name}, skipping.")
                filtered_count += 1
                continue
            
            if vocals_energy < energy_threshold_db:
                print(f"Filtered out {base_name}: vocals_energy={vocals_energy:.2f}dB")
                # Remove both files
                if os.path.exists(vocals_path):
                    os.remove(vocals_path)
                if os.path.exists(instrumental_src):
                    os.remove(instrumental_src)
                filtered_count += 1
                continue
            
            # Move and resample instrumental file
            if os.path.exists(instrumental_src):
                if resample_and_save(instrumental_src, instrumental_dst, target_sr=16000):
                    os.remove(instrumental_src)
                    moved_count += 1
                else:
                    print(f"Warning: Resampling failed for {instrumental_src}, moving directly.")
                    shutil.move(instrumental_src, instrumental_dst)
                    moved_count += 1

            # Resample vocals file in place
            if os.path.exists(vocals_path):
                temp_vocals = vocals_path + ".tmp.wav"
                if resample_and_save(vocals_path, temp_vocals, target_sr=16000):
                    shutil.move(temp_vocals, vocals_path)
                
                txt_path = os.path.join(store_dir, f"{base_name}_vocals.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(utterance_text)
                txt_count += 1
            
    print(f"\nPost-processing complete.")
    print(f"Filtered out {filtered_count} samples due to low energy (< {energy_threshold_db} dB).")
    print(f"Moved {moved_count} instrumental files to {ins_dir}")
    print(f"Created {txt_count} text files.")

if __name__ == "__main__":
    main()
