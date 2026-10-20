import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from PIL import ImageFile

from model import WasteClassifier

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_NAMES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TRAIN_NAMES = {"train", "training"}
VAL_NAMES = {"val", "valid", "validation"}
TEST_NAMES = {"test", "testing"}


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_name(name):
    return "".join(ch.lower() for ch in name if ch.isalnum())


def class_dir_map(root):
    dirs = [p for p in root.iterdir() if p.is_dir()]
    mapping = {normalized_name(p.name): p for p in dirs}
    aliases = {
        "cardboard": ["cardboard", "carton"],
        "glass": ["glass"],
        "metal": ["metal"],
        "paper": ["paper"],
        "plastic": ["plastic"],
        "trash": ["trash", "garbage", "other", "waste"],
    }
    selected = {}
    for canonical, names in aliases.items():
        for alias in names:
            if alias in mapping:
                selected[canonical] = mapping[alias]
                break
    if len(selected) == 6:
        return [selected[normalized_name(c)] for c in CLASS_NAMES]
    return None


def has_images(folder):
    return folder.exists() and any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        for p in folder.rglob("*")
    )


def find_split(root, names):
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() in names and class_dir_map(p):
            return p
    return None


def make_imagefolder(root, transform):
    class_dirs = class_dir_map(root)
    if class_dirs is None:
        raise RuntimeError(
            f"Could not find all six class folders under: {root}\n"
            f"Expected folders named Cardboard, Glass, Metal, Paper, Plastic, Trash "
            f"(capitalization does not matter)."
        )
    # Build a dataset with fixed class index order, independent of alphabetical ordering.
    samples = []
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    for label, folder in enumerate(class_dirs):
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(p), label))
    if not samples:
        raise RuntimeError(f"No images found under {root}.")
    ds = datasets.ImageFolder.__new__(datasets.ImageFolder)
    ds.root = str(root)
    ds.loader = datasets.folder.default_loader
    ds.extensions = tuple(IMAGE_EXTENSIONS)
    ds.classes = CLASS_NAMES
    ds.class_to_idx = class_to_idx
    ds.samples = samples
    ds.imgs = samples
    ds.targets = [label for _, label in samples]
    ds.transform = transform
    ds.target_transform = None
    return ds


def count_labels(dataset):
    counts = [0] * len(CLASS_NAMES)
    for _, label in dataset.samples:
        counts[label] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Train the compact six-class waste classifier.")
    parser.add_argument("--data", default="data", help="Dataset root folder (default: data)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation fraction when no validation folder exists")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0,
                        help="Use 0 on Windows for the most reliable startup")
    parser.add_argument("--patience", type=int, default=6,
                        help="Early stopping patience; 0 disables early stopping")
    args = parser.parse_args()

    if not 0.05 <= args.val_split <= 0.5:
        raise ValueError("--val-split must be between 0.05 and 0.5")

    seed_everything(args.seed)
    data_root = Path(args.data)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset folder '{data_root}' was not found. Put your dataset in that folder "
            "or run: python train.py --data PATH_TO_DATASET"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Class order:", CLASS_NAMES)

    train_transform = transforms.Compose([
        transforms.Resize((104, 104)),
        transforms.RandomResizedCrop(96, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_root = find_split(data_root, TRAIN_NAMES)
    val_root = find_split(data_root, VAL_NAMES)
    test_root = find_split(data_root, TEST_NAMES)

    if train_root:
        train_ds = make_imagefolder(train_root, train_transform)
        if val_root:
            val_ds = make_imagefolder(val_root, eval_transform)
            print(f"Using existing training folder: {train_root}")
            print(f"Using existing validation folder: {val_root}")
        else:
            # If a test split exists, don't use it as validation.
            base_eval_ds = make_imagefolder(train_root, eval_transform)
            n_val = max(1, int(len(base_eval_ds) * args.val_split))
            n_train = len(base_eval_ds) - n_val
            if n_train < 1:
                raise RuntimeError("Not enough images to create training and validation subsets.")
            generator = torch.Generator().manual_seed(args.seed)
            train_indices, val_indices = random_split(range(len(base_eval_ds)), [n_train, n_val], generator=generator)
            train_ds = Subset(train_ds, train_indices.indices)
            val_ds = Subset(base_eval_ds, val_indices.indices)
            print("No validation folder found; split the training folder into train/validation.")
    else:
        # Single root with class folders: create a deterministic random split.
        all_train = make_imagefolder(data_root, train_transform)
        all_eval = make_imagefolder(data_root, eval_transform)
        n_val = max(1, int(len(all_train) * args.val_split))
        n_train = len(all_train) - n_val
        if n_train < 1:
            raise RuntimeError("Not enough images to create training and validation subsets.")
        generator = torch.Generator().manual_seed(args.seed)
        train_indices, val_indices = random_split(range(len(all_train)), [n_train, n_val], generator=generator)
        train_ds = Subset(all_train, train_indices.indices)
        val_ds = Subset(Subset(all_eval, list(range(len(all_eval)))), val_indices.indices)
        print(f"No train/validation split detected; random {1-args.val_split:.0%}/{args.val_split:.0%} split.")

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Training or validation dataset is empty.")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory)

    model = WasteClassifier(num_classes=6).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {parameter_count:,}")
    if parameter_count >= 350_000:
        raise RuntimeError(f"Model has {parameter_count:,} parameters; limit is fewer than 350,000.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pth"
    best_acc = -1.0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_n = labels.size(0)
            train_loss_sum += loss.item() * batch_n
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += batch_n

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                batch_n = labels.size(0)
                val_loss_sum += loss.item() * batch_n
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += batch_n

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.2%} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "input_size": [1, 3, 96, 96],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "best_val_accuracy": best_acc,
                "parameter_count": parameter_count,
            }, best_path)
            print(f"  Saved new best checkpoint: {best_path}")
        else:
            stale_epochs += 1

        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping: validation accuracy did not improve for {args.patience} epochs.")
            break

    print(f"\nBest validation accuracy: {best_acc:.2%}")
    print(f"Best checkpoint: {best_path}")
    print("Next step: python export.py")


if __name__ == "__main__":
    main()
