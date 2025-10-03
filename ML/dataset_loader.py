import json
import os
import numpy as np

def load_level_json(path):
    """
    Expects JSON files with {"grid": [[ints]]}
    """
    with open(path, 'r') as f:
        j = json.load(f)
    return np.array(j['grid'], dtype=np.int64)

def load_dataset_from_dir(dirpath, max_files=1000):
    files = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith('.json')]
    files = files[:max_files]
    data = []
    for p in files:
        data.append(load_level_json(p))
    return data

if __name__ == "__main__":
    print("Example: run load_dataset_from_dir on local dataset")
