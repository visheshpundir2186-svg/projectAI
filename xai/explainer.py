import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def swin_reshape_transform(tensor):
    """
    Reshape transform for Swin Transformer in pytorch-grad-cam.
    Converts (B, H, W, C) to (B, C, H, W) for CAM heatmap generation.
    """
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    return tensor

def get_target_layer_for_gradcam(model, model_name: str):
    """Identifies the optimal convolutional / projection layer for Grad-CAM."""
    name = model_name.lower()
    if name == "attention_unet":
        return [model.bottleneck.conv[3]], None
    elif name == "deeplabv3":
        return [model.aspp.project[0]], None
    elif name == "segformer":
        return [model.model.segformer.encoder.block[3][-1].norm1], None
    elif name in ["swin", "swin_transformer"]:
        # Target final stage of Swin features
        return [model.model.features[-1]], swin_reshape_transform
    elif name in ["resnet", "resnet50"]:
        # Target final conv block of layer4 in ResNet-50
        return [model.model.layer4[-1]], None
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_gradcam(model, model_name: str, input_tensor, raw_image_np):
    """
    Computes Grad-CAM heatmap using pytorch-grad-cam.
    raw_image_np: float array in [0, 1] with shape (H, W, 3)
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        target_layers, reshape_fn = get_target_layer_for_gradcam(model, model_name)
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_fn)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_image = show_cam_on_image(raw_image_np, grayscale_cam, use_rgb=True)
        return cam_image
    except Exception as e:
        print(f"Grad-CAM note: {e}")
        return (raw_image_np * 255).astype(np.uint8)

def run_lime(model, raw_image_np, device, num_samples=300):
    """
    Computes LIME superpixel importance mask.
    """
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries

        explainer = lime_image.LimeImageExplainer()

        def batch_predict(images):
            tensors = torch.from_numpy(images).permute(0, 3, 1, 2).float()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            norm_tensors = (tensors - mean) / std

            model.eval()
            with torch.no_grad():
                logits = model(norm_tensors.to(device))
                probs = F.softmax(logits, dim=-1)
            return probs.cpu().numpy()

        explanation = explainer.explain_instance(
            raw_image_np.astype(np.double),
            batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        lime_vis = mark_boundaries(temp, mask)
        return (lime_vis * 255).astype(np.uint8)
    except Exception as e:
        print(f"LIME note: {e}")
        return (raw_image_np * 255).astype(np.uint8)

def generate_xai_comparison(model, model_name: str, input_tensor, raw_image_np, class_name: str, save_path="results/xai_comparison.png"):
    """
    Generates a high-quality multi-panel figure comparing Original, Grad-CAM, and LIME.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = input_tensor.device

    print(f"Generating Grad-CAM for {model_name}...")
    cam_img = run_gradcam(model, model_name, input_tensor, raw_image_np)

    print(f"Generating LIME for {model_name}...")
    lime_img = run_lime(model, raw_image_np, device)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(raw_image_np)
    axes[0].set_title(f"Input Satellite Tile\nTarget: {class_name}", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(cam_img)
    axes[1].set_title(f"Grad-CAM Heatmap\n({model_name.upper()})", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(lime_img)
    axes[2].set_title(f"LIME Superpixel Regions\n({model_name.upper()})", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved XAI comparison plot to: {save_path}")
