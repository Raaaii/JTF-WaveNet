import os
import shutil

source_root = "."
target_root = "data_proc/organized_fids"
os.makedirs(target_root, exist_ok=True)

for i in range(480, 491):
    src = os.path.join(source_root, str(i), "fid_phased.fid")
    dst_dir = os.path.join(target_root, f"vd{i}")
    dst = os.path.join(dst_dir, "fid_phased.fid")
    
    os.makedirs(dst_dir, exist_ok=True)
    
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
    else:
        print(f"[WARNING] Missing: {src}")
