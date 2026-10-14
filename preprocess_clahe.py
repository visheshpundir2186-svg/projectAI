import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from utils.clahe import apply_clahe

def process_dataset_with_clahe(input_dir="data", output_dir="data_clahe", clip_limit=2.0):
    """
    Applies CLAHE enhancement across all images in the dataset splits
    (train, val, test) and saves them into output_dir.
    """
    print(f"Starting CLAHE preprocessing from '{input_dir}' to '{output_dir}'...")
    total_processed = 0
    sample_before = None
    sample_after = None

    for split in ["train", "val", "test"]:
        split_in = os.path.join(input_dir, split)
        split_out = os.path.join(output_dir, split)

        if not os.path.exists(split_in):
            continue

        for cls_name in os.listdir(split_in):
            cls_in = os.path.join(split_in, cls_name)
            cls_out = os.path.join(split_out, cls_name)

            if not os.path.isdir(cls_in):
                continue

            os.makedirs(cls_out, exist_ok=True)

            for fname in os.listdir(cls_in):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    in_path = os.path.join(cls_in, fname)
                    out_path = os.path.join(cls_out, fname)

                    raw_img = Image.open(in_path).convert("RGB")
                    clahe_img = apply_clahe(raw_img, clip_limit=clip_limit)
                    clahe_img.save(out_path)

                    if sample_before is None:
                        sample_before = raw_img
                        sample_after = clahe_img

                    total_processed += 1

    print(f"CLAHE preprocessing complete! Total images processed: {total_processed}")

    # Generate a before-and-after comparison plot
    if sample_before is not None and sample_after is not None:
        os.makedirs("results", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(sample_before)
        axes[0].set_title("Original Satellite Tile", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(sample_after)
        axes[1].set_title("CLAHE Preprocessed\n(Enhanced Local Contrast)", fontsize=12)
        axes[1].axis("off")

        cmp_path = "results/clahe_comparison_sample.png"
        plt.tight_layout()
        plt.savefig(cmp_path, dpi=300)
        plt.close()
        print(f"Saved before/after CLAHE comparison visual to: {cmp_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset images with CLAHE algorithm")
    parser.add_argument("--input_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="data_clahe")
    parser.add_argument("--clip_limit", type=float, default=2.0)
    args = parser.parse_args()

    process_dataset_with_clahe(args.input_dir, args.output_dir, args.clip_limit)
