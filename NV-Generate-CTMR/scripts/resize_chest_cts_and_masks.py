from pathlib import Path
import argparse
import re
import shutil

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def resize_nifti(in_path: Path, out_path: Path, target_shape, order: int, make_binary_mask: bool = False):
    """
    Resize a NIfTI file to target_shape.

    order=1:
        linear interpolation, use for CT images.

    order=0:
        nearest-neighbor interpolation, use for masks.

    make_binary_mask=True:
        convert resized mask to 0/1 after resizing.
    """
    img = nib.load(str(in_path))
    arr = img.get_fdata()

    old_shape = np.array(arr.shape[:3], dtype=np.float32)
    new_shape = np.array(target_shape, dtype=np.float32)

    zoom_factors = new_shape / old_shape
    resized = zoom(arr, zoom_factors, order=order)

    # Update affine so physical field-of-view is approximately preserved.
    affine = img.affine.copy()
    scale = old_shape / new_shape
    affine[:3, :3] = affine[:3, :3] @ np.diag(scale)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if make_binary_mask:
        resized = (resized > 0).astype(np.uint8)
    elif order == 0:
        resized = resized.astype(np.uint8)
    else:
        resized = resized.astype(np.float32)

    nib.save(nib.Nifti1Image(resized, affine), str(out_path))


def find_ct_file(patient_dir: Path, masks_dir_name: str):
    """
    Find the main CT NIfTI directly inside the patient folder.

    Excludes files that look like masks/labels/segmentations.
    If multiple CT candidates exist, picks the largest file.
    """
    candidates = []

    for p in patient_dir.iterdir():
        if not p.is_file():
            continue

        if not (p.name.endswith(".nii.gz") or p.name.endswith(".nii")):
            continue

        low = p.name.lower()

        if masks_dir_name.lower() in low:
            continue

        if any(x in low for x in ["mask", "label", "seg", "roi"]):
            continue

        candidates.append(p)

    if len(candidates) == 0:
        raise FileNotFoundError(f"No CT NIfTI found directly under {patient_dir}")

    candidates = sorted(candidates, key=lambda x: x.stat().st_size, reverse=True)

    if len(candidates) > 1:
        print(f"[WARN] multiple CT candidates in {patient_dir}, using largest: {candidates[0].name}")

    return candidates[0]


def find_text_file(patient_dir: Path):
    """
    Find the text file directly inside the patient folder.

    If multiple .txt files exist, picks the first alphabetically.
    """
    txts = sorted([p for p in patient_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])

    if len(txts) == 0:
        raise FileNotFoundError(f"No .txt file found directly under {patient_dir}")

    if len(txts) > 1:
        print(f"[WARN] multiple text files in {patient_dir}, using: {txts[0].name}")

    return txts[0]


def count_nonempty_text_lines(txt_path: Path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return len([line for line in f.readlines() if line.strip()])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        required=True,
        help="Original dataset root containing patient folders.",
    )

    parser.add_argument(
        "--out-root",
        required=True,
        help="Output root. Will contain the same patient-folder structure as raw-root.",
    )

    parser.add_argument(
        "--masks-dir-name",
        default="masks",
        help="Name of mask folder inside each patient folder.",
    )

    parser.add_argument(
        "--target-shape",
        nargs=3,
        type=int,
        default=[256, 256, 128],
        help="Target resized shape, e.g. 256 256 128.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()
    target_shape = tuple(args.target_shape)

    if not raw_root.exists():
        raise FileNotFoundError(f"raw-root does not exist: {raw_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])

    if len(patient_dirs) == 0:
        raise RuntimeError(f"No patient directories found under {raw_root}")

    print("[INFO] raw_root:", raw_root)
    print("[INFO] out_root:", out_root)
    print("[INFO] target_shape:", target_shape)
    print("[INFO] masks_dir_name:", args.masks_dir_name)
    print("[INFO] patients:", len(patient_dirs))

    for patient_dir in patient_dirs:
        patient = patient_dir.name

        in_masks_dir = patient_dir / args.masks_dir_name
        out_patient_dir = out_root / patient
        out_masks_dir = out_patient_dir / args.masks_dir_name

        if not in_masks_dir.exists():
            print(f"[SKIP] no masks dir: {patient_dir}")
            continue

        ct_path = find_ct_file(patient_dir, args.masks_dir_name)
        txt_path = find_text_file(patient_dir)

        mask_paths = sorted(
            [
                p
                for p in in_masks_dir.iterdir()
                if p.is_file() and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))
            ],
            key=natural_key,
        )

        if len(mask_paths) == 0:
            print(f"[SKIP] no mask files: {in_masks_dir}")
            continue

        n_text_lines = count_nonempty_text_lines(txt_path)

        if n_text_lines != len(mask_paths):
            raise ValueError(
                f"{patient}: number of non-empty text lines ({n_text_lines}) "
                f"does not match number of masks ({len(mask_paths)}). "
                f"text={txt_path}, masks={in_masks_dir}"
            )

        out_patient_dir.mkdir(parents=True, exist_ok=True)
        out_masks_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # Resize CT
        # -------------------------
        out_ct_path = out_patient_dir / ct_path.name

        if out_ct_path.exists() and not args.overwrite:
            print(f"[SKIP CT exists] {out_ct_path}")
        else:
            print(f"[CT] {patient}: {ct_path.name} -> {out_ct_path}")
            resize_nifti(
                in_path=ct_path,
                out_path=out_ct_path,
                target_shape=target_shape,
                order=1,
                make_binary_mask=False,
            )

        # -------------------------
        # Copy text file unchanged
        # -------------------------
        out_txt_path = out_patient_dir / txt_path.name

        if out_txt_path.exists() and not args.overwrite:
            print(f"[SKIP TXT exists] {out_txt_path}")
        else:
            print(f"[TXT] {patient}: {txt_path.name} -> {out_txt_path}")
            shutil.copy2(txt_path, out_txt_path)

        # -------------------------
        # Resize masks
        # -------------------------
        for mask_path in mask_paths:
            out_mask_path = out_masks_dir / mask_path.name

            if out_mask_path.exists() and not args.overwrite:
                print(f"[SKIP MASK exists] {out_mask_path}")
                continue

            print(f"[MASK] {patient}: {mask_path.name} -> {out_mask_path}")

            resize_nifti(
                in_path=mask_path,
                out_path=out_mask_path,
                target_shape=target_shape,
                order=0,
                make_binary_mask=True,
            )

        print(f"[DONE] {patient}: CT=1, masks={len(mask_paths)}, text_lines={n_text_lines}")

    print("\n[DONE ALL]")
    print("Resized dataset written to:")
    print(out_root)
    print("\nThis output has the same folder format as the original dataset.")
    print("Next stage: run your prepare_chest_text_controlnet_dataset.py script using this resized dataset as --raw-root.")


if __name__ == "__main__":
    main()