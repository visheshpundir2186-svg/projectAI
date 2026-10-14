import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(img_size=(224, 224)):
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform

def get_dataloaders(data_dir="data", batch_size=16, img_size=(224, 224), num_workers=0):
    train_tf, eval_tf = get_transforms(img_size)

    train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_set = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_tf)
    test_set = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_set.classes
