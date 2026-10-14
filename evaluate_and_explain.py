import os
import argparse
import torch
import numpy as np
from PIL import Image

from train import build_model
from utils.dataset import get_dataloaders
from xai.explainer import generate_xai_comparison

def main():
    parser = argparse.ArgumentParser(description="Generate XAI Explanations for Groundwater Models")
    parser.add_argument("--model", type=str, default="swin", choices=["swin", "segformer", "attention_unet", "deeplabv3", "resnet"])
    parser.add_argument("--data_dir", type=str, default="data_clahe")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sample_image", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, classes = get_dataloaders(args.data_dir, batch_size=1)
    num_classes = len(classes)

    model = build_model(args.model, num_classes=num_classes).to(device)
    ckpt_path = args.checkpoint or f"checkpoints/best_{args.model}.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"Notice: Checkpoint '{ckpt_path}' not found. Using initialized model for demonstration.")
    model.eval()

    # Pick sample from test_loader
    for images, labels in test_loader:
        input_tensor = images.to(device)
        class_name = classes[labels[0].item()]
        
        # Un-normalize image for plotting
        img_np = images[0].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        raw_img = np.clip(img_np * std + mean, 0.0, 1.0)
        break

    save_path = f"results/xai_{args.model}_{class_name}.png"
    generate_xai_comparison(
        model=model,
        model_name=args.model,
        input_tensor=input_tensor,
        raw_image_np=raw_img,
        class_name=class_name,
        save_path=save_path
    )

if __name__ == "__main__":
    main()
