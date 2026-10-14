import os
import json
import argparse
import matplotlib.pyplot as plt
from train import train_single_model, ALL_MODELS

def train_and_compare_all(data_dir="data_clahe", epochs=10, batch_size=16):
    """
    Trains all 5 models sequentially on the CLAHE-enhanced dataset
    and produces a benchmark summary table and comparative graph.
    """
    print("==================================================================")
    print(" TRAIN ALL 5 ALGORITHMS ON CLAHE-ENHANCED GROUNDWATER DATASET")
    print(" 1. Swin Transformer (Shifted Windows Vision Transformer)")
    print(" 2. SegFormer (Hierarchical Mix Transformer)")
    print(" 3. Attention U-Net (Attention Gate CNN)")
    print(" 4. DeepLabV3+ (Atrous Spatial Pyramid Pooling)")
    print(" 5. ResNet-50 (Residual Network Baseline)")
    print("==================================================================")

    # Learning rate customization per architecture class
    lrs = {
        "swin": 1e-4,
        "segformer": 5e-5,
        "attention_unet": 3e-4,
        "deeplabv3": 1e-4,
        "resnet": 3e-4
    }

    results = []
    os.makedirs("results", exist_ok=True)

    for model_name in ALL_MODELS:
        res = train_single_model(
            model_name=model_name,
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lrs.get(model_name, 1e-4)
        )
        results.append(res)

    # Sort results by Macro F1 descending
    results = sorted(results, key=lambda x: x["test_f1"], reverse=True)

    # Print Leaderboard
    print("\n" + "="*65)
    print(f"{'Rank':<6}{'Algorithm':<18}{'Test Acc':<12}{'Macro F1':<12}{'Test Loss':<10}")
    print("="*65)
    for rank, r in enumerate(results, 1):
        print(f"{rank:<6}{r['model'].upper():<18}{r['test_acc']*100:<11.2f}%{r['test_f1']*100:<11.2f}%{r['test_loss']:<10.4f}")
    print("="*65)

    # Save JSON report
    with open("results/five_algorithms_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate Comparative Bar Chart
    models = [r['model'].upper() for r in results]
    accs = [r['test_acc'] * 100 for r in results]
    f1s = [r['test_f1'] * 100 for r in results]

    x = range(len(models))
    plt.figure(figsize=(10, 6))
    plt.bar([i - 0.2 for i in x], accs, width=0.4, label="Accuracy (%)", color="#3b82f6")
    plt.bar([i + 0.2 for i in x], f1s, width=0.4, label="Macro F1 (%)", color="#10b981")
    plt.xticks(x, models)
    plt.ylabel("Score (%)")
    plt.title("Groundwater Evaluation: 5 Algorithms Comparison (with CLAHE Preprocessing)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    chart_path = "results/five_algorithms_benchmark.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved benchmark graph to: {chart_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all five algorithms on CLAHE preprocessed data")
    parser.add_argument("--data_dir", type=str, default="data_clahe")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train_and_compare_all(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
