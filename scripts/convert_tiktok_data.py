#!/usr/bin/env python3
"""
scripts/convert_tiktok_data.py
==============================
Data Adapter Script: Converts DiffMM's TikTok dataset (ACM MM 2024)
into the standard format required by STAIR (AAAI 2025).

Input:
  DiffMM/Datasets/tiktok/
    - trnMat.pkl, valMat.pkl, tstMat.pkl (scipy.sparse.coo_matrix)
    - audio_feat.npy, image_feat.npy, text_feat.npy

Output:
  data/tiktok/
    - Raw files copied for reference
    - train.txt, valid.txt, test.txt (USER \t ITEM \t TIMESTAMP)
    - visual_modality.pkl (torch.Tensor [6710, 128], float32)
    - textual_modality.pkl (torch.Tensor [6710, 768], float32)
    - audio_modality.pkl (torch.Tensor [6710, 128], float32)
    - multimodal_concat.pkl (torch.Tensor [6710, 1024], float32 - Concatenated audio+image+text)
"""

import os
import shutil
import pickle
import numpy as np
import torch
import scipy.sparse as sp

def convert_tiktok(
    src_dir: str = "DiffMM/Datasets/tiktok",
    dst_dir: str = "data/tiktok"
):
    print(f"=== Converting TikTok dataset from {src_dir} to {dst_dir} ===")
    os.makedirs(dst_dir, exist_ok=True)

    # 1. Copy raw files
    raw_files = [
        "trnMat.pkl", "valMat.pkl", "tstMat.pkl",
        "audio_feat.npy", "image_feat.npy", "text_feat.npy"
    ]
    for rf in raw_files:
        src_path = os.path.join(src_dir, rf)
        dst_path = os.path.join(dst_dir, rf)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  [Copied] {rf} -> {dst_path}")
        else:
            raise FileNotFoundError(f"Missing source file: {src_path}")

    # 2. Load interaction matrices
    trn_mat = pickle.load(open(os.path.join(src_dir, "trnMat.pkl"), "rb"))
    val_mat = pickle.load(open(os.path.join(src_dir, "valMat.pkl"), "rb"))
    tst_mat = pickle.load(open(os.path.join(src_dir, "tstMat.pkl"), "rb"))

    num_users, num_items = trn_mat.shape
    print(f"\nInteraction matrix dimensions: {num_users} Users x {num_items} Items")
    print(f"Train NNZ: {trn_mat.nnz:,} | Valid NNZ: {val_mat.nnz:,} | Test NNZ: {tst_mat.nnz:,}")

    # 3. Export interaction txt files (USER \t ITEM \t TIMESTAMP)
    splits = [
        ("train.txt", trn_mat),
        ("valid.txt", val_mat),
        ("test.txt", tst_mat)
    ]

    for fname, mat in splits:
        out_path = os.path.join(dst_dir, fname)
        # Sort by user_id ascending, then item_id ascending
        rows = mat.row
        cols = mat.col
        order = np.lexsort((cols, rows))
        sorted_rows = rows[order]
        sorted_cols = cols[order]

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("USER\tITEM\tTIMESTAMP\n")
            for u, i in zip(sorted_rows, sorted_cols):
                f.write(f"{u}\t{i}\t0\n")
        print(f"  [Generated] {fname}: {len(sorted_rows):,} interactions written to {out_path}")

    # 4. Load and process multimodal feature vectors
    audio_feat = np.load(os.path.join(src_dir, "audio_feat.npy"))
    image_feat = np.load(os.path.join(src_dir, "image_feat.npy"))
    text_feat  = np.load(os.path.join(src_dir, "text_feat.npy"))

    print(f"\nMultimodal feature shapes:")
    print(f"  Audio: {audio_feat.shape} ({audio_feat.dtype})")
    print(f"  Image: {image_feat.shape} ({image_feat.dtype})")
    print(f"  Text : {text_feat.shape} ({text_feat.dtype})")

    # Convert to float32 PyTorch tensors
    audio_tensor = torch.from_numpy(audio_feat.astype(np.float32))
    image_tensor = torch.from_numpy(image_feat.astype(np.float32))
    text_tensor  = torch.from_numpy(text_feat.astype(np.float32))

    # Multimodal concatenation (audio + image + text)
    concat_np = np.concatenate([audio_feat, image_feat, text_feat], axis=1)
    concat_tensor = torch.from_numpy(concat_np.astype(np.float32))
    print(f"  Concatenated: {concat_tensor.shape} ({concat_tensor.dtype})")

    # Save modalities as PyTorch pickle objects expected by STAIR
    modality_maps = {
        "visual_modality.pkl": image_tensor,
        "textual_modality.pkl": text_tensor,
        "audio_modality.pkl": audio_tensor,
        "multimodal_concat.pkl": concat_tensor
    }

    for pkl_name, tensor_data in modality_maps.items():
        out_path = os.path.join(dst_dir, pkl_name)
        with open(out_path, "wb") as f:
            pickle.dump(tensor_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  [Generated] {pkl_name}: {tensor_data.shape} ({size_mb:.2f} MB)")

    print(f"\n[SUCCESS] TikTok dataset conversion complete! Files ready in {dst_dir}/")

if __name__ == "__main__":
    convert_tiktok()
