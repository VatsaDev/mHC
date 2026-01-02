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
dataset_path = "math"
DATA_CACHE_DIR = dataset_path
shard_size = 10_000_000  
num_workers = 8         
chunk_read_size = 5 * 1024 * 1024 # 5MB of text per task

tokenizer = None

def init_worker(path):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)

def tokenize_worker(text_chunk):
    """Tokenizes a chunk of text sent from the main process."""
    try:
        return tokenizer.encode(text_chunk, add_special_tokens=False)
    except Exception as e:
        return []

def write_datafile(filename, data_shard):
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520        
    header[1] = 1               
    header[2] = len(data_shard) 
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(data_shard.tobytes())

def stream_chunks(files):
    """Yields text chunks from files to keep memory usage low."""
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                # Read chunks by size, but finish the current line to avoid splitting words
                chunk = f.read(chunk_read_size)
                if not chunk:
                    break
                extra = f.readline() 
                yield chunk + extra

def process_split(files, split_name):
    print(f"Processing {split_name} split...")
    token_buffer = []
    shard_count = 0
    
    with mp.Pool(num_workers, initializer=init_worker, initargs=(tok_path,)) as pool:
        # We don't know total chunks in advance, so we use a simple counter pbar
        pbar = tqdm(desc=f"Tokenizing {split_name}")
        
        for ids in pool.imap(tokenize_worker, stream_chunks(files)):
            token_buffer.extend(ids)
            
            while len(token_buffer) >= shard_size:
                shard_data = np.array(token_buffer[:shard_size], dtype=np.uint16)
                filename = os.path.join(DATA_CACHE_DIR, f"{dataset_path}_{split_name}_{shard_count:06d}.bin")
                write_datafile(filename, shard_data)
                shard_count += 1
                token_buffer = token_buffer[shard_size:]
            pbar.update(1)
        pbar.close()

    if token_buffer:
        shard_data = np.array(token_buffer, dtype=np.uint16)
        filename = os.path.join(DATA_CACHE_DIR, f"{dataset_path}_{split_name}_{shard_count:06d}.bin")
        write_datafile(filename, shard_data)

if __name__ == "__main__":
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    input_files = sorted(glob.glob(os.path.join(dataset_path, "*.txt")))
    
    if not input_files:
        print(f"Error: No .txt files found in {dataset_path}")
        exit()

    n = len(input_files)
    split_idx = max(1, int(n * 0.9))
    train_files = input_files[:split_idx]
    val_files = input_files[split_idx:]

    process_split(train_files, "train")
    process_split(val_files, "val")

    tok = AutoTokenizer.from_pretrained(tok_path)
    meta = {'vocab_size': tok.vocab_size, 'tokenizer_path': tok_path, 'vocab_source': 'custom_bpe'}
    with open(os.path.join(DATA_CACHE_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    gitignore_path = '../.gitignore'
    entry = f"{dataset_path}/\n"
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            if entry not in f.readlines():
                with open(gitignore_path, 'a') as f: f.write(entry)
    else:
        with open(gitignore_path, 'w') as f: f.write(entry)

    print(f"Done! Dataset ready in: {DATA_CACHE_DIR}")
