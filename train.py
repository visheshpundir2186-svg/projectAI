import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score

from utils.dataset import get_dataloaders
from models.attention_unet import AttentionUNetClassifier
from models.deeplabv3_classifier import DeepLabV3PlusClassifier
from models.segformer_classifier import SegFormerClassifier
from models.swin_classifier import SwinTransformerClassifier
from models.resnet_classifier import ResNet50Classifier

ALL_MODELS = ["swin", "segformer", "attention_unet", "deeplabv3", "resnet"]

def build_model(model_name: str, num_classes: int):
    name = model_name.lower()
    if name == "attention_unet":
        return AttentionUNetClassifier(num_classes=num_classes)
    elif name == "deeplabv3":
        return DeepLabV3PlusClassifier(num_classes=num_classes)
    elif name == "segformer":
        return SegFormerClassifier(num_classes=num_classes)
    elif name in ["swin", "swin_transformer"]:
        return SwinTransformerClassifier(num_classes=num_classes)
    elif name in ["resnet", "resnet50"]:
        return ResNet50Classifier(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from {ALL_MODELS}")

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1

def train_single_model(model_name: str, data_dir="data_clahe", epochs=15, batch_size=16, lr=1e-4, weight_decay=1e-4, optimizer_type="adamw"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Training: {model_name.upper()} on CLAHE Dataset ({data_dir})")
    print(f" Device: {device} | Epochs: {epochs} | LR: {lr}")
    print(f"=======================================================")

    train_loader, val_loader, test_loader, classes = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(classes)

    model = build_model(model_name, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    os.makedirs("checkpoints", exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {tr_loss:.4f} | Acc: {tr_acc:.3f} | F1: {tr_f1:.3f} || "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.3f} | F1: {val_f1:.3f}")

        if val_f1 >= best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"checkpoints/best_{model_name}.pth")

    # Final test evaluation
    model.load_state_dict(torch.load(f"checkpoints/best_{model_name}.pth"))
    test_loss, test_acc, test_f1 = eval_epoch(model, test_loader, criterion, device)
    print(f">> Final Test Result for {model_name.upper()}: Acc = {test_acc:.4f} | Macro F1 = {test_f1:.4f} | Loss = {test_loss:.4f}")

    return {
        "model": model_name,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_loss": test_loss,
        "best_val_f1": best_f1
    }

def main():
    parser = argparse.ArgumentParser(description="Train Image Models for Groundwater Resource Evaluation")
    parser.add_argument("--model", type=str, default="swin", choices=ALL_MODELS)
    parser.add_argument("--data_dir", type=str, default="data_clahe")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    args = parser.parse_args()

    train_single_model(
        model_name=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer
    )

if __name__ == "__main__":
    main()
