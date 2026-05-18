import argparse
import pandas as pd
import numpy as np
import os
import subprocess
import warnings

warnings.filterwarnings('ignore')

# --- 保持工具函数不变 ---
def parse_time_vectorized(series):
    s = series.str.strip().str.replace(',', '.', regex=False)
    dt = pd.to_datetime(s, format='%H:%M:%S.%f', errors='coerce')
    seconds = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second + dt.dt.microsecond / 1e6
    return seconds.fillna(0.0)

def normalize_text(text):
    if text is None: return ""
    try:
        return str(text).encode("latin1", errors="strict").decode("cp1252")
    except:
        return str(text)

def check_duplicate_start(text1, text2):
    if not text1 or not text2: return False
    t1 = str(text1).strip().split()
    t2 = str(text2).strip().split()
    if len(t1) < 2 or len(t2) < 2: return False
    return (t1[0], t1[1]) == (t2[0], t2[1])

# --- V2C Core Algorithm Adaptation ---

def is_sequence_valid(records, start_idx, length, max_gap=10.0):
    prev = records[start_idx]
    for k in range(1, length):
        curr = records[start_idx + k]
        # Same Dialogue and Speaker
        if curr['Dialogue_ID'] != prev['Dialogue_ID'] or curr['Speaker'] != prev['Speaker']:
            return False
        # Continuous Utterance ID (ensure no skipped lines)
        if curr['Utterance_ID'] != prev['Utterance_ID'] + 1:
            return False
        # Time gap
        if (curr['Start_Sec'] - prev['End_Sec']) > max_gap:
            return False
        prev = curr
    return True

def has_emotion_change(records, start_idx, length):
    emotions = set(records[start_idx + k]['Emotion'] for k in range(length))
    return len(emotions) > 1

def check_word_count(records, start_idx, length, max_words=50):
    total = sum(len(str(records[start_idx + k]['Utterance_Clean']).split()) for k in range(length))
    return total <= max_words

def build_sample(records, index, length):
    rows = [records[index + k] for k in range(length)]
    first, last = rows[0], rows[-1]
    
    sample = {
        'Speaker': first['Speaker'],
        'Dialogue_ID': first['Dialogue_ID'],
        'Season': first.get('Season'),
        'Episode': first.get('Episode'),
        'Emotions': [r['Emotion'] for r in rows],
        'Utterances': [r['Utterance_Clean'] for r in rows],
        'Start_Time': first['StartTime'], 
        'End_Time': last['EndTime'],
        'Length': length,
        'Utterance_IDs': [r['Utterance_ID'] for r in rows],
        'Line_Indices': ",".join([str(r.get('Sr No.', '')) for r in rows]),
        # Helper for deduplication
        '_temp_utt_ids': tuple(r['Utterance_ID'] for r in rows)
    }
    return sample

def find_optimal_samples(file_path, output_path, video_input_dir=None, video_output_dir=None, prefer_longer=False):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 1. Preprocessing
    if 'Dialogue_ID' in df.columns and 'Utterance_ID' in df.columns:
        df = df.sort_values(by=['Dialogue_ID', 'Utterance_ID']).reset_index(drop=True)
    
    df['Utterance_Clean'] = df['Utterance'].apply(normalize_text)
    
    if 'StartTime' in df.columns:
        df['Start_Sec'] = parse_time_vectorized(df['StartTime'])
        df['End_Sec'] = parse_time_vectorized(df['EndTime'])
    else:
        print("Error: Missing time columns.")
        return

    records = df.to_dict('records')
    n = len(records)
    valid_samples = []
    
    print(f"Scanning {n} rows... Strategy: Sliding Window (V2C Algorithm).")
    
    # Core Sliding Window Loop
    for i in range(n - 1):
        found_length_3 = False
        
        # Check length 3 first (when prefer_longer is enabled)
        if i < n - 2:
            if is_sequence_valid(records, i, 3):
                if has_emotion_change(records, i, 3):
                    if check_word_count(records, i, 3):
                        valid_samples.append(build_sample(records, i, 3))
                        found_length_3 = True
        
        # Check length 2 (skip if prefer_longer and found length 3)
        if not (prefer_longer and found_length_3):
            if is_sequence_valid(records, i, 2):
                if has_emotion_change(records, i, 2):
                    valid_samples.append(build_sample(records, i, 2))

    # Output
    result_df = pd.DataFrame(valid_samples)
    print(f"Total valid samples (before deduplication): {len(result_df)}")
    
    if not result_df.empty:
        # Deduplication based on Dialogue_ID and Utterance IDs (to avoid cross-dialogue collisions)
        result_df = result_df.loc[result_df.astype(str).drop_duplicates(subset=['Dialogue_ID', '_temp_utt_ids']).index]
        result_df = result_df.drop(columns=['_temp_utt_ids'])
        print(f"Total valid samples (after deduplication): {len(result_df)}")

        # Columns
        desired_cols = ['Sr No.', 'Length', 'Speaker', 'Emotions', 'Start_Time', 'End_Time', 'Utterances']
        final_cols = [c for c in desired_cols if c in result_df.columns] + \
                     [c for c in result_df.columns if c not in desired_cols]
        
        result_df = result_df[final_cols]
        result_df.to_csv(output_path, index=False)
        print(f"Saved clean CSV to {output_path}")

        if video_input_dir and video_output_dir:
            process_video_samples(result_df, video_input_dir, video_output_dir)

def process_video_samples(df, input_dir, output_dir):
    # (此处保持不变，使用 df['Start_Time'] 和 df['End_Time'] 无需修改)
    # 注意：你的原逻辑是按文件合并(concat)，不需要时间戳。
    # 如果你是要按时间切割，才需要 Start/End。
    # 既然原来是合并文件，这里保留原来的合并逻辑即可。
    print(f"\nProcessing videos for {len(df)} samples...")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    for idx, row in df.iterrows():
        # 这里原来的逻辑是根据 Utterance_IDs 找文件，不受 Start_Time 字段影响
        # 所以这里不需要改动
        utt_ids = row['Utterance_IDs']
        dia_id = row['Dialogue_ID']
        
        files = []
        possible = True
        for uid in utt_ids:
            fname = f"dia{dia_id}_utt{uid}.mp4"
            fpath = os.path.join(input_dir, fname)
            if os.path.exists(fpath):
                files.append(fpath)
            else:
                possible = False; break
        
        if possible and files:
            list_path = os.path.join(output_dir, f"list_{idx}.txt")
            out_path = os.path.join(output_dir, f"sample_{idx}.mp4")
            with open(list_path, 'w') as f:
                for fp in files: f.write(f"file '{fp}'\n")
            subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', '-y', out_path], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(list_path): os.remove(list_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='dev_sent_emo.csv')
    parser.add_argument('--output', default='dev.csv')
    parser.add_argument('--video-input-dir')
    parser.add_argument('--video-output-dir')
    parser.add_argument('--prefer-longer', action='store_true', help='If length 3 sample is valid, skip length 2 at the same position')
    args = parser.parse_args()

    find_optimal_samples(args.input, args.output, 
                         video_input_dir=args.video_input_dir, 
                         video_output_dir=args.video_output_dir,
                         prefer_longer=args.prefer_longer)