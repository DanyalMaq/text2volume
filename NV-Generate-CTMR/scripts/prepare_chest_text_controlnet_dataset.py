from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import monai
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose


def natural_key(path: Path):
    """
    Sort mask1, mask2, ..., mask10 in human order.
    """
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def strip_nii_gz(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def round_number(number: int, base_number: int = 128) -> int:
    """
    Same logic as repo's diff_model_create_training_data.py:
    round each spatial dimension to nearest multiple of 128, minimum 128.
    """
    new_number = max(round(float(number) / float(base_number)), 1.0) * float(base_number)
    return int(new_number)


def find_ct_file(patient_dir: Path, masks_dir_name: str) -> Path:
    """
    Finds the main CT NIfTI directly under patient_dir.
    Excludes files inside masks/ and excludes obvious mask/label names.
    """
    candidates = []
    for p in patient_dir.iterdir():
        if not p.is_file():
            continue
        if not (p.name.endswith(".nii.gz") or p.name.endswith(".nii")):
            continue
        lowered = p.name.lower()
        if any(x in lowered for x in ["mask", "label", "seg"]):
            continue
        candidates.append(p)

    if len(candidates) == 0:
        raise FileNotFoundError(f"No CT NIfTI found directly under {patient_dir}")

    if len(candidates) > 1:
        # Usually the CT is the largest NIfTI.
        candidates = sorted(candidates, key=lambda x: x.stat().st_size, reverse=True)
        print(f"[WARN] multiple CT candidates in {patient_dir}, using largest: {candidates[0].name}")

    return candidates[0]


def find_text_file(patient_dir: Path) -> Path:
    candidates = sorted([p for p in patient_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    if len(candidates) == 0:
        raise FileNotFoundError(f"No .txt file found directly under {patient_dir}")
    if len(candidates) > 1:
        print(f"[WARN] multiple txt files in {patient_dir}, using: {candidates[0].name}")
    return candidates[0]


def read_text_lines(txt_path: Path) -> list[str]:
    lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def get_ct_new_dim(ct_path: Path) -> tuple[int, int, int]:
    """
    Replicates the repo's embedding-size decision:
    load image, ensure channel-first, orient RAS, then round dims to multiples of 128.
    """
    tfm = Compose(
        [
            monai.transforms.LoadImaged(keys="image"),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
        ]
    )
    d = tfm({"image": str(ct_path)})
    img = d["image"]
    dim = [int(img.meta["dim"][i]) for i in range(1, 4)]
    new_dim = tuple(round_number(x, 128) for x in dim)
    return new_dim


def make_label_transform(new_dim: tuple[int, int, int]) -> Compose:
    """
    Match the label spatial grid to the preprocessed CT grid used before autoencoder encoding.

    Important:
      - mask resize uses nearest neighbor
      - output remains integer label
      - only the mask is needed, not full segmentation
    """
    return Compose(
        [
            monai.transforms.LoadImaged(keys="label"),
            monai.transforms.EnsureChannelFirstd(keys="label"),
            monai.transforms.Orientationd(keys="label", axcodes="RAS"),
            monai.transforms.EnsureTyped(keys="label", dtype=torch.float32, track_meta=True),
            monai.transforms.Resized(keys="label", spatial_size=new_dim, mode="nearest"),
        ]
    )


def save_combined_label(
    mask_path: Path,
    out_path: Path,
    new_dim: tuple[int, int, int],
    target_label: int,
):
    """
    Converts one binary/nonzero ROI mask into an integer label map:
      0 = everything else
      target_label = described ROI
    """
    tfm = make_label_transform(new_dim)
    d = tfm({"label": str(mask_path)})
    mask = d["label"]

    affine = mask.meta["affine"].cpu().numpy() if hasattr(mask.meta["affine"], "cpu") else mask.meta["affine"]

    arr = mask.detach().cpu().numpy()
    arr = np.squeeze(arr)

    combined = np.zeros(arr.shape, dtype=np.uint8)
    combined[arr > 0] = np.uint8(target_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(combined, affine=affine), str(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, help="Root containing patient folders.")
    parser.add_argument("--out-root", required=True, help="Processed dataset root.")
    parser.add_argument("--masks-dir-name", default="masks")
    parser.add_argument("--target-label", type=int, default=23)
    parser.add_argument("--modality", default="ct")
    parser.add_argument("--fold-train", type=int, default=1)
    parser.add_argument("--fold-val", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--top-region-index", nargs=4, type=int, default=[0, 1, 0, 0])
    parser.add_argument("--bottom-region-index", nargs=4, type=int, default=[0, 1, 0, 0])
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()

    labels_root = out_root / "labels"
    embeddings_root = out_root / "embeddings"

    raw_image_entries = []
    train_entries = []

    patient_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    if len(patient_dirs) == 0:
        raise RuntimeError(f"No patient directories found under {raw_root}")

    n_val_patients = int(round(len(patient_dirs) * args.val_frac))
    val_patient_names = set([p.name for p in patient_dirs[:n_val_patients]])

    print(f"[INFO] Found {len(patient_dirs)} patient dirs")
    print(f"[INFO] Validation patients: {len(val_patient_names)}")
    print(f"[INFO] Target ROI label: {args.target_label}")

    for patient_dir in patient_dirs:
        patient_name = patient_dir.name
        masks_dir = patient_dir / args.masks_dir_name

        if not masks_dir.is_dir():
            print(f"[WARN] missing masks dir, skipping: {patient_dir}")
            continue

        ct_path = find_ct_file(patient_dir, args.masks_dir_name)
        txt_path = find_text_file(patient_dir)
        text_lines = read_text_lines(txt_path)

        mask_paths = sorted(
            [p for p in masks_dir.iterdir() if p.name.endswith(".nii.gz") or p.name.endswith(".nii")],
            key=natural_key,
        )

        if len(mask_paths) == 0:
            print(f"[WARN] no masks found, skipping: {patient_dir}")
            continue

        if len(text_lines) != len(mask_paths):
            raise ValueError(
                f"{patient_name}: number of text lines ({len(text_lines)}) "
                f"does not match number of masks ({len(mask_paths)}). "
                f"txt={txt_path}, masks_dir={masks_dir}"
            )

        # Path to CT relative to raw root, for repo embedding script.
        ct_rel_raw = ct_path.relative_to(raw_root).as_posix()
        raw_image_entries.append(
            {
                "image": ct_rel_raw,
                "modality": args.modality,
            }
        )

        new_dim = get_ct_new_dim(ct_path)
        print(f"[INFO] {patient_name}: CT={ct_path.name}, new_dim={new_dim}, masks={len(mask_paths)}")

        # The repo embedding script will save:
        #   embeddings_root / patient / ct_stem_emb.nii.gz
        ct_stem = strip_nii_gz(ct_path.name)
        emb_rel_to_processed = Path("embeddings") / patient_name / f"{ct_stem}_emb.nii.gz"

        fold = args.fold_val if patient_name in val_patient_names else args.fold_train

        for idx, (mask_path, text) in enumerate(zip(mask_paths, text_lines), start=1):
            mask_stem = strip_nii_gz(mask_path.name)
            label_rel_to_processed = Path("labels") / patient_name / f"{mask_stem}_combined_label.nii.gz"
            label_out = out_root / label_rel_to_processed

            save_combined_label(
                mask_path=mask_path,
                out_path=label_out,
                new_dim=new_dim,
                target_label=args.target_label,
            )

            train_entries.append(
                {
                    "image": emb_rel_to_processed.as_posix(),
                    "label": label_rel_to_processed.as_posix(),
                    "text": text,
                    "dim": list(new_dim),
                    "spacing": [1.0, 1.0, 1.0],
                    "top_region_index": args.top_region_index,
                    "bottom_region_index": args.bottom_region_index,
                    "modality": args.modality,
                    "fold": fold,
                    "patient_id": patient_name,
                    "mask_name": mask_path.name,
                }
            )

    out_root.mkdir(parents=True, exist_ok=True)

    raw_json = out_root / "raw_image_list_for_embeddings.json"
    train_json = out_root / "chest_text_controlnet_train.json"

    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump({"training": raw_image_entries}, f, indent=2)

    with open(train_json, "w", encoding="utf-8") as f:
        json.dump({"training": train_entries}, f, indent=2)

    print("\n[DONE]")
    print(f"Raw image JSON for embedding script: {raw_json}")
    print(f"Final ControlNet training JSON:      {train_json}")
    print(f"Labels root:                         {labels_root}")
    print(f"Embeddings root expected at:         {embeddings_root}")
    print(f"Number of CTs:                       {len(raw_image_entries)}")
    print(f"Number of mask/text training items:  {len(train_entries)}")
    print("\nNext step: run scripts.diff_model_create_training_data to create the embeddings.")


if __name__ == "__main__":
    main()

# python -m scripts.resize_chest_cts_and_masks \
#   --raw-root datasets/rexgrounding_original \
#   --out-root datasets/rexgrounding \
#   --masks-dir-name masks \
#   --target-shape 256 256 128

# python -m scripts.prepare_chest_text_controlnet_dataset \
#   --raw-root datasets/rexgrounding \
#   --out-root datasets/rexgrounding_processed \
#   --masks-dir-name masks \
#   --target-label 23 \
#   --val-frac 0.1

# python -m scripts.diff_model_create_training_data \
#    -e configs/environment_create_chest_text_embeddings.json \
#    -c configs/config_maisi_diff_model_rflow-ct.json \
#    -t configs/config_network_rflow_text.json \
#    -g 1

# Train
# python -m scripts.train_controlnet \
#   -t configs/config_network_rflow_text.json \
#   -c configs/config_maisi_controlnet_train_chest_text.json \
#   -e configs/environment_maisi_controlnet_train_chest_text.json \
#   -g 1

# MultiGPU Train
# export NUM_GPUS_PER_NODE=3
# torchrun \
#   --nproc_per_node=${NUM_GPUS_PER_NODE} \
#   --nnodes=1 \
#   --master_addr=localhost \
#   --master_port=1234 \
#   -m scripts.train_controlnet \
#   -t configs/config_network_rflow_text.json \
#   -c configs/config_maisi_controlnet_train_chest_text.json \
#   -e configs/environment_maisi_controlnet_train_chest_text.json \
#   -g ${NUM_GPUS_PER_NODE}

# Inference
# python -m scripts.inference \
#   -t configs/config_network_rflow_text.json \
#   -e configs/environment_maisi_infer_chest_text.json \
#   -i configs/config_infer_chest_text.json \
#   --version rflow-ct

# Validate
# python - <<'PY'
# import json
# import os
# from pathlib import Path
# import nibabel as nib
# import numpy as np

# base = Path("datasets/rexgrounding_processed")
# json_path = base / "chest_text_controlnet_train.json"

# d = json.load(open(json_path))
# items = d["training"]

# print("num items:", len(items))

# missing = 0
# for i, item in enumerate(items):
#     image = base / item["image"]
#     label = base / item["label"]

#     if not image.exists():
#         print("[MISSING EMB]", image)
#         missing += 1
#     if not label.exists():
#         print("[MISSING LABEL]", label)
#         missing += 1

#     if i < 5 and image.exists() and label.exists():
#         lab = nib.load(str(label)).get_fdata()
#         print(i, image.name, label.name, item["text"][:80])
#         print("  label shape:", lab.shape, "unique:", np.unique(lab)[:10])

# print("missing:", missing)
# PY