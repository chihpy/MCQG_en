"""
"""
import os
import json

def mkdir(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

def json_load(file_path):
    with open(file_path, 'r') as f:
        data_dict = json.load(f)
    return data_dict

def json_dump(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print(f"Data successfully written to {file_path}")
