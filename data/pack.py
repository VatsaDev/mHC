# pack text into bin files

import os
import glob
import pickle
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Setup & Hyperparameters ---
tok_path = "../tokenizers/mhc_1k" 
dataset_path = "synth_text"
DATA_CACHE_DIR = dataset_path
shard_size = 10_000_000  # 10 million tokens per shard
num_workers = 1         # Use 10 cores

# Global tokenizer variable for workers (initialized once per process)
tokenizer = None

def init_worker(path):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)

def tokenize_worker(file_path):
    """Worker function: reads a file and returns a list of token IDs."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Add special tokens (BOS/EOS) as per your original script
        return tokenizer.encode(text, add_special_tokens=True)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def write_datafile(filename, data_shard):
    """Writes a shard with the specific 256-byte header format."""
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520        # magic
    header[1] = 1               # version
    header[2] = len(data_shard) # number of tokens
    
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(data_shard.tobytes())

def process_split(files, split_name):
    """Streams tokens from workers and writes shards when buffer is full."""
    print(f"Processing {len(files)} files for {split_name} split...")
    
    token_buffer = []
    shard_count = 0
    
    # Use imap to stream results back as soon as a worker finishes a file
    with mp.Pool(num_workers, initializer=init_worker, initargs=(tok_path,)) as pool:
        pbar = tqdm(total=len(files), desc=f"Tokenizing {split_name}")
        
        for ids in pool.imap(tokenize_worker, files):
            token_buffer.extend(ids)
            
            # If buffer exceeds shard_size, write out shards
            while len(token_buffer) >= shard_size:
                shard_data = np.array(token_buffer[:shard_size], dtype=np.uint16)
                filename = os.path.join(DATA_CACHE_DIR, f"{dataset_path}_{split_name}_{shard_count:06d}.bin")
                write_datafile(filename, shard_data)
                
                shard_count += 1
                token_buffer = token_buffer[shard_size:] # Remove written tokens
            
            pbar.update(1)
        pbar.close()

    # Write any remaining tokens as the final (smaller) shard
    if token_buffer:
        shard_data = np.array(token_buffer, dtype=np.uint16)
        filename = os.path.join(DATA_CACHE_DIR, f"{dataset_path}_{split_name}_{shard_count:06d}.bin")
        write_datafile(filename, shard_data)

if __name__ == "__main__":
    # 1. Setup Directory
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    input_files = sorted(glob.glob(os.path.join(dataset_path, "*.txt")))
    
    if not input_files:
        print(f"Error: No .txt files found in {dataset_path}")
        exit()

    # 2. Train/Val Split (90/10)
    n = len(input_files)
    split_idx = int(n * 0.9)
    train_files = input_files[:split_idx]
    val_files = input_files[split_idx:]

    # 3. Process each split
    process_split(train_files, "train")
    process_split(val_files, "val")

    # 4. Save Meta Information
    # Initialize tokenizer once in main to get vocab size
    tok = AutoTokenizer.from_pretrained(tok_path)
    meta = {
        'vocab_size': tok.vocab_size,
        'tokenizer_path': tok_path, 
        'vocab_source': 'custom_bpe' 
    }
    with open(os.path.join(DATA_CACHE_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    # 5. Gitignore Update
    gitignore_path = '../.gitignore'
    entry = f"{dataset_path}/\n"
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            if entry not in f.readlines():
                with open(gitignore_path, 'a') as f:
                    f.write(entry)
    else:
        with open(gitignore_path, 'w') as f:
            f.write(entry)

    print(f"Done! Dataset ready in: {DATA_CACHE_DIR}")
