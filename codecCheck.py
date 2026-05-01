import pandas as pd
from pathlib import Path
import torch

# Paths
TSV_TRACK2 = r"C:\Users\Deepal\Desktop\College\SEM VI\MAJOR PROJECT FINAL\ASVspoof5.dev.track_2.trial.tsv"
DEV_DIR    = r"D:\dev_spectrograms"

print("--- 1. Checking the TSV ---")
try:
    with open(TSV_TRACK2, "r") as f:
        print("First 5 lines of Track 2 TSV:")
        lines = [next(f) for _ in range(5)]
        for line in lines:
            print(f"  {line.strip()}")
            
        # Try to parse the first line manually
        parts = lines[0].strip().split()
        if len(parts) >= 2:
            print(f"\nExample ID from TSV: '{parts[1]}'")
        else:
            print("\nError: TSV format is strange. Not enough columns.")
except Exception as e:
    print(f"Could not read TSV: {e}")

print("\n--- 2. Checking the DEV Directory ---")
pt_files = list(Path(DEV_DIR).glob("*.pt"))
if not pt_files:
    print("No .pt files found in DEV_DIR!")
else:
    print(f"Found {len(pt_files)} .pt files.")
    print("First 5 files:")
    for file in pt_files[:5]:
        print(f"  {file.name}")
        
    print("\nLoading the first .pt file to inspect 'file_ids'...")
    try:
        d = torch.load(pt_files[0], map_location="cpu")
        if "file_ids" in d:
             file_ids = d["file_ids"]
             print("First 5 'file_ids' stored INSIDE the .pt file:")
             for fid in file_ids[:5]:
                 print(f"  {fid}")
        else:
             print("WARNING: The .pt file does NOT contain a 'file_ids' key. The model doesn't know what these files are.")
    except Exception as e:
        print(f"Could not load .pt file: {e}")

print("\n--- End of Diagnostic ---")