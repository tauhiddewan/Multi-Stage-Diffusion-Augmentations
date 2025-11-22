import os
import pickle
from pathlib import Path
import random
from dotenv import dotenv_values

def collect_image_mask_pairs(images_dir, masks_dir):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    img_files = {p.stem: p for p in images_dir.iterdir() if p.is_file()}
    mask_files = {p.stem: p for p in masks_dir.iterdir() if p.is_file()}

    common_stems = sorted(set(img_files.keys()) & set(mask_files.keys()))
    pairs = [(str(img_files[s]), str(mask_files[s])) for s in common_stems]
    return pairs


def main():
    env_vars = dotenv_values(dotenv_path="./.env")
    pickle_filename = str(env_vars.get("split_fname", "train_test_dataset.pkl"))

    data_root = "./data"          
    pickle_filename = "dataset.pkl" 

    id_folder = "kvsir"             # in-distribution (Kvasir-SEG)
    ood1_folder = "clinicdb"        # OOD1
    ood2_folder = "colondb"         # OOD2

    train_size = 800
    val_size = 100
    test_size = 100
    # ================================

    # ---- Collect all pairs ----
    id_pairs = collect_image_mask_pairs(
        images_dir=os.path.join(data_root, id_folder, "images"),
        masks_dir=os.path.join(data_root, id_folder, "masks"),
    )

    ood1_pairs = collect_image_mask_pairs(
        images_dir=os.path.join(data_root, ood1_folder, "images"),
        masks_dir=os.path.join(data_root, ood1_folder, "masks"),
    )

    ood2_pairs = collect_image_mask_pairs(
        images_dir=os.path.join(data_root, ood2_folder, "images"),
        masks_dir=os.path.join(data_root, ood2_folder, "masks"),
    )

    # ---- Shuffle & split ID dataset ----
    random.seed(123)
    random.shuffle(id_pairs)

    train_data = id_pairs[:train_size]
    val_data = id_pairs[train_size : train_size+val_size]
    test_data_id = id_pairs[train_size+val_size : train_size+val_size+test_size]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test(ID): {len(test_data_id)}")
    print(f"OOD1 (clinicdb): {len(ood1_pairs)}, OOD2 (colondb): {len(ood2_pairs)}")

    data = {
        "train_data": train_data,
        "val_data": val_data,
        "test_data_id": test_data_id,      # in-distribution test set
        "test_data_ood1": ood1_pairs,        # OOD1: cvc-clinicdb
        "test_data_ood2": ood2_pairs,        # OOD2: cvc-colondb
    }

    out_path = os.path.join(data_root, pickle_filename)

    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved splits pickle to: {out_path}")


if __name__ == "__main__":
    main()
