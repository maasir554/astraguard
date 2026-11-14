# ============================================================
# CELL 1: Install & Imports
# ============================================================
!pip install -q segmentation-models-pytorch
!pip install -q albumentations
!pip install -q rasterio geopandas shapely

import os, glob, random, warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import shape

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# CELL 2: Download SpaceNet 7 (subset for Colab)
# ============================================================
# Option A: If you have a local copy, upload/mount from Drive
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_ROOT = '/content/drive/MyDrive/spacenet7'

# Option B: Download a sample from AWS (SpaceNet 7 is public on S3)
# Full dataset is ~100 GB; we download a manageable subset here.

!pip install -q awscli

# Create directory structure
!mkdir -p /content/spacenet7/train/images
!mkdir -p /content/spacenet7/train/labels

# Download a few Area-of-Interest (AOI) folders
# SpaceNet 7 structure: s3://spacenet-dataset/spacenet/SN7_buildings/
# Each AOI has: images/ (monthly GeoTIFFs) and labels_match/ (GeoJSON)

import subprocess

AOIS = [
    "L15-0331E-1257N_1327_3160_13",
    "L15-0357E-1223N_1429_3296_13",
    "L15-0368E-1245N_1474_3208_13",
    "L15-0434E-1218N_1736_3316_13",
    "L15-0487E-1246N_1950_3204_13",
]

S3_BASE = "s3://spacenet-dataset/spacenet/SN7_buildings/train"

for aoi in tqdm(AOIS, desc="Downloading AOIs"):
    # Download images
    subprocess.run([
        "aws", "s3", "cp",
        f"{S3_BASE}/{aoi}/images_masked/",
        f"/content/spacenet7/train/{aoi}/images/",
        "--recursive", "--no-sign-request", "--quiet"
    ])
    # Download labels
    subprocess.run([
        "aws", "s3", "cp",
        f"{S3_BASE}/{aoi}/labels_match/",
        f"/content/spacenet7/train/{aoi}/labels/",
        "--recursive", "--no-sign-request", "--quiet"
    ])

DATA_ROOT = "/content/spacenet7/train"
print(f"Data root: {DATA_ROOT}")
print(f"AOIs downloaded: {len(os.listdir(DATA_ROOT))}")


# ============================================================
# CELL 3: SpaceNet 7 Building Segmentation Dataset
# ============================================================

class SpaceNet7BuildingDataset(Dataset):
    """
    Reads SpaceNet 7 images + GeoJSON labels.
    Rasterizes building polygons into binary masks on-the-fly.
    """
    def __init__(self, data_root, aois=None, transform=None, img_size=512):
        self.transform = transform
        self.img_size = img_size
        self.samples = []  # list of (image_path, geojson_path)

        aoi_dirs = aois if aois else os.listdir(data_root)
        for aoi in aoi_dirs:
            img_dir = os.path.join(data_root, aoi, "images")
            lbl_dir = os.path.join(data_root, aoi, "labels")
            if not os.path.isdir(img_dir):
                continue

            for img_file in sorted(glob.glob(os.path.join(img_dir, "*.tif"))):
                # Match image to its GeoJSON label
                base = os.path.basename(img_file)
                # SpaceNet 7 naming: global_monthly_YYYY_MM_mosaic_{aoi}.tif
                # Labels: SN7_buildings_train_{aoi}_YYYY_MM.geojson
                parts = base.replace(".tif", "").split("_")
                # Try to find the matching geojson
                geojsons = glob.glob(os.path.join(lbl_dir, "*.geojson"))
                # Simple matching by date substring
                year_month = None
                for p in parts:
                    if len(p) == 4 and p.isdigit():
                        idx = parts.index(p)
                        if idx + 1 < len(parts):
                            year_month = f"{p}_{parts[idx+1]}"
                            break

                matched_lbl = None
                if year_month:
                    for gj in geojsons:
                        if year_month in gj:
                            matched_lbl = gj
                            break
                if matched_lbl is None and len(geojsons) > 0:
                    # Fallback: use the geojson with closest name
                    matched_lbl = geojsons[0]

                if matched_lbl:
                    self.samples.append((img_file, matched_lbl))

        print(f"Found {len(self.samples)} image-label pairs")

    def _rasterize_geojson(self, geojson_path, reference_img_path):
        """Rasterize building polygons to a binary mask."""
        with rasterio.open(reference_img_path) as src:
            transform = src.transform
            h, w = src.height, src.width

        try:
            gdf = gpd.read_file(geojson_path)
            if len(gdf) == 0:
                return np.zeros((h, w), dtype=np.uint8)

            shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and geom.is_valid]
            if len(shapes) == 0:
                return np.zeros((h, w), dtype=np.uint8)

            mask = rasterize(
                shapes,
                out_shape=(h, w),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )
            return mask
        except Exception:
            return np.zeros((h, w), dtype=np.uint8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # Read image (may have 3+ bands)
        with rasterio.open(img_path) as src:
            img = src.read()  # (C, H, W)
            # Take first 3 bands (RGB)
            if img.shape[0] >= 3:
                img = img[:3]
            else:
                img = np.repeat(img, 3, axis=0)[:3]

        img = img.transpose(1, 2, 0).astype(np.float32)  # (H, W, C)

        # Normalize to [0, 1]
        for c in range(3):
            cmin, cmax = img[:,:,c].min(), img[:,:,c].max()
            if cmax > cmin:
                img[:,:,c] = (img[:,:,c] - cmin) / (cmax - cmin)

        # Rasterize mask
        mask = self._rasterize_geojson(lbl_path, img_path)  # (H, W)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1))
            mask = torch.from_numpy(mask)

        mask = mask.float().unsqueeze(0)  # (1, H, W)
        return img, mask


# ============================================================
# CELL 4: Augmentations & DataLoaders
# ============================================================
IMG_SIZE = 512
BATCH_SIZE = 4

train_transform = A.Compose([
    A.RandomCrop(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Build dataset
full_dataset = SpaceNet7BuildingDataset(
    DATA_ROOT,
    aois=AOIS,
    transform=None  # We'll apply transforms after split
)

# Split 80/20
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Wrapper to apply different transforms per split
class TransformWrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img_path, lbl_path = self.subset.dataset.samples[self.subset.indices[idx]]
        with rasterio.open(img_path) as src:
            img = src.read()[:3].transpose(1, 2, 0).astype(np.float32)
            for c in range(3):
                cmin, cmax = img[:,:,c].min(), img[:,:,c].max()
                if cmax > cmin:
                    img[:,:,c] = (img[:,:,c] - cmin) / (cmax - cmin)
        mask = self.subset.dataset._rasterize_geojson(lbl_path, img_path)
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1))
            mask = torch.from_numpy(mask)
        return img, mask.float().unsqueeze(0)

train_ds = TransformWrapper(train_subset, train_transform)
val_ds   = TransformWrapper(val_subset, val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


# ============================================================
# CELL 5: Model Definitions (U-Net & DeepLabV3+)
# ============================================================

# --- Model A: U-Net with ResNet34 encoder ---
unet_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,  # We'll use BCEWithLogitsLoss
)

# --- Model B: DeepLabV3+ with ResNet50 encoder ---
deeplabv3plus_model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,
)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"U-Net params:        {count_params(unet_model):,}")
print(f"DeepLabV3+ params:   {count_params(deeplabv3plus_model):,}")


# ============================================================
# CELL 6: Loss Functions & Metrics
# ============================================================

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for binary segmentation."""
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def compute_iou(preds, targets, threshold=0.5):
    """Compute IoU (Intersection over Union)."""
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def compute_f1(preds, targets, threshold=0.5):
    """Compute pixel-wise F1 score."""
    preds = (torch.sigmoid(preds) > threshold).float()
    tp = (preds * targets).sum(dim=(2, 3))
    fp = (preds * (1 - targets)).sum(dim=(2, 3))
    fn = ((1 - preds) * targets).sum(dim=(2, 3))
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1.mean().item()


# ============================================================
# CELL 7: Training Function
# ============================================================

def train_model(model, model_name, train_loader, val_loader,
                num_epochs=30, lr=1e-4, device=DEVICE):
    model = model.to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_iou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': []}

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs} Train"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_f1 = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs} Val"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_iou += compute_iou(outputs, masks)
                val_f1 += compute_f1(outputs, masks)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_f1 /= len(val_loader)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_f1'].append(val_f1)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), f"/content/{model_name}_best.pth")
            print(f"  ✅ Saved best {model_name} (IoU: {best_val_iou:.4f})")

    return model, history


# ============================================================
# CELL 8: Train U-Net
# ============================================================
print("=" * 60)
print("Training U-Net")
print("=" * 60)

unet_model, unet_history = train_model(
    unet_model, "UNet",
    train_loader, val_loader,
    num_epochs=30, lr=1e-4
)


# ============================================================
# CELL 9: Train DeepLabV3+
# ============================================================
print("=" * 60)
print("Training DeepLabV3+")
print("=" * 60)

deeplabv3plus_model, dlv3_history = train_model(
    deeplabv3plus_model, "DeepLabV3Plus",
    train_loader, val_loader,
    num_epochs=30, lr=1e-4
)


# ============================================================
# CELL 10: Plot Training Curves & Predictions
# ============================================================

def plot_training_curves(histories, names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for hist, name in zip(histories, names):
        axes[0].plot(hist['train_loss'], label=f'{name} Train')
        axes[0].plot(hist['val_loss'], '--', label=f'{name} Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)

    for hist, name in zip(histories, names):
        axes[1].plot(hist['val_iou'], label=name)
    axes[1].set_title('Validation IoU'); axes[1].legend(); axes[1].grid(True)

    for hist, name in zip(histories, names):
        axes[2].plot(hist['val_f1'], label=name)
    axes[2].set_title('Validation F1'); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('/content/model3_training_curves.png', dpi=150)
    plt.show()

plot_training_curves(
    [unet_history, dlv3_history],
    ['U-Net', 'DeepLabV3+']
)


def visualize_predictions(model, model_name, dataloader, n=4):
    model.eval()
    imgs, masks = next(iter(dataloader))
    imgs = imgs.to(DEVICE)
    with torch.no_grad():
        preds = torch.sigmoid(model(imgs)) > 0.5

    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    for i in range(min(n, imgs.shape[0])):
        img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title('Input Image')
        axes[i, 1].imshow(masks[i, 0].cpu(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 2].imshow(preds[i, 0].cpu(), cmap='gray')
        axes[i, 2].set_title(f'{model_name} Prediction')
        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'/content/{model_name}_predictions.png', dpi=150)
    plt.show()

visualize_predictions(unet_model, "U-Net", val_loader)
visualize_predictions(deeplabv3plus_model, "DeepLabV3+", val_loader)


# ============================================================
# CELL 11: Save Final Models
# ============================================================

# Save to Google Drive (optional)
# import shutil
# shutil.copy('/content/UNet_best.pth', '/content/drive/MyDrive/UNet_best.pth')
# shutil.copy('/content/DeepLabV3Plus_best.pth', '/content/drive/MyDrive/DeepLabV3Plus_best.pth')

# Print final metrics
print("\n" + "=" * 60)
print("MODEL 3 — BUILDING SEGMENTATION RESULTS")
print("=" * 60)
print(f"U-Net       — Best IoU: {max(unet_history['val_iou']):.4f} | "
      f"Best F1: {max(unet_history['val_f1']):.4f}")
print(f"DeepLabV3+  — Best IoU: {max(dlv3_history['val_iou']):.4f} | "
      f"Best F1: {max(dlv3_history['val_f1']):.4f}")
