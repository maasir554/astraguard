# ============================================================
# CELL 1: Install & Imports
# ============================================================
!pip install -q segmentation-models-pytorch
!pip install -q albumentations
!pip install -q rasterio geopandas einops timm

import os, glob, random, warnings, itertools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import timm

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# CELL 2: Download SpaceNet 7 temporal imagery
# ============================================================
!pip install -q awscli

!mkdir -p /content/spacenet7/train

import subprocess

# SpaceNet 7 contains monthly images per AOI — perfect for change detection
AOIS = [
    "L15-0331E-1257N_1327_3160_13",
    "L15-0357E-1223N_1429_3296_13",
    "L15-0368E-1245N_1474_3208_13",
    "L15-0434E-1218N_1736_3316_13",
    "L15-0487E-1246N_1950_3204_13",
]

S3_BASE = "s3://spacenet-dataset/spacenet/SN7_buildings/train"

for aoi in tqdm(AOIS, desc="Downloading AOIs"):
    subprocess.run([
        "aws", "s3", "cp",
        f"{S3_BASE}/{aoi}/images_masked/",
        f"/content/spacenet7/train/{aoi}/images/",
        "--recursive", "--no-sign-request", "--quiet"
    ])
    subprocess.run([
        "aws", "s3", "cp",
        f"{S3_BASE}/{aoi}/labels_match/",
        f"/content/spacenet7/train/{aoi}/labels/",
        "--recursive", "--no-sign-request", "--quiet"
    ])

DATA_ROOT = "/content/spacenet7/train"
print(f"Downloaded {len(AOIS)} AOIs for temporal change detection")


# ============================================================
# CELL 3: Bi-Temporal Change Detection Dataset
# ============================================================

class SpaceNet7ChangeDataset(Dataset):
    """
    Creates bi-temporal image pairs from SpaceNet 7 monthly imagery.
    Change mask = XOR of building masks at time T1 and T2.

    Each sample: (img_t1, img_t2, change_mask)
    where change_mask = 1 where buildings appeared or disappeared.
    """
    def __init__(self, data_root, aois, transform=None, img_size=512,
                 min_month_gap=3):
        self.transform = transform
        self.img_size = img_size
        self.pairs = []  # (img_t1, lbl_t1, img_t2, lbl_t2)

        for aoi in aois:
            img_dir = os.path.join(data_root, aoi, "images")
            lbl_dir = os.path.join(data_root, aoi, "labels")
            if not os.path.isdir(img_dir):
                continue

            # Collect all timestamped images
            imgs = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
            lbls = sorted(glob.glob(os.path.join(lbl_dir, "*.geojson")))

            if len(imgs) < 2 or len(lbls) < 2:
                continue

            # Match images to labels by index (they should be sorted same)
            n = min(len(imgs), len(lbls))

            # Create pairs with temporal gap
            for i in range(n):
                for j in range(i + min_month_gap, n):
                    self.pairs.append((imgs[i], lbls[i], imgs[j], lbls[j]))

        # Limit to manageable size
        if len(self.pairs) > 2000:
            random.seed(42)
            self.pairs = random.sample(self.pairs, 2000)

        print(f"Created {len(self.pairs)} bi-temporal pairs")

    def _read_image(self, path):
        with rasterio.open(path) as src:
            img = src.read()
            if img.shape[0] >= 3:
                img = img[:3]
            else:
                img = np.repeat(img, 3, axis=0)[:3]
        img = img.transpose(1, 2, 0).astype(np.float32)
        for c in range(3):
            cmin, cmax = img[:,:,c].min(), img[:,:,c].max()
            if cmax > cmin:
                img[:,:,c] = (img[:,:,c] - cmin) / (cmax - cmin)
        return img

    def _rasterize_label(self, geojson_path, ref_img_path):
        with rasterio.open(ref_img_path) as src:
            transform = src.transform
            h, w = src.height, src.width
        try:
            gdf = gpd.read_file(geojson_path)
            if len(gdf) == 0:
                return np.zeros((h, w), dtype=np.uint8)
            shapes = [(g, 1) for g in gdf.geometry if g is not None and g.is_valid]
            if not shapes:
                return np.zeros((h, w), dtype=np.uint8)
            return rasterize(shapes, out_shape=(h, w), transform=transform,
                           fill=0, dtype=np.uint8)
        except Exception:
            return np.zeros((h, w), dtype=np.uint8)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, lbl1_path, img2_path, lbl2_path = self.pairs[idx]

        img1 = self._read_image(img1_path)
        img2 = self._read_image(img2_path)
        mask1 = self._rasterize_label(lbl1_path, img1_path)
        mask2 = self._rasterize_label(lbl2_path, img2_path)

        # Change mask: XOR (new buildings OR demolished buildings)
        change_mask = ((mask1 == 0) & (mask2 == 1)) | ((mask1 == 1) & (mask2 == 0))
        change_mask = change_mask.astype(np.uint8)

        # Apply augmentations to both images consistently
        if self.transform:
            # Augment both images the same way (spatial transforms)
            seed = random.randint(0, 2**32)
            random.seed(seed); np.random.seed(seed)
            aug1 = self.transform(image=img1, mask=change_mask)
            random.seed(seed); np.random.seed(seed)
            aug2 = self.transform(image=img2, mask=change_mask)

            img1 = aug1['image']
            img2 = aug2['image']
            change_mask = aug1['mask']
        else:
            img1 = torch.from_numpy(img1.transpose(2, 0, 1))
            img2 = torch.from_numpy(img2.transpose(2, 0, 1))
            change_mask = torch.from_numpy(change_mask)

        change_mask = change_mask.float().unsqueeze(0)
        return img1, img2, change_mask


# ============================================================
# CELL 4: DataLoaders
# ============================================================
IMG_SIZE = 256  # Smaller for change detection (two images per sample)
BATCH_SIZE = 4

train_transform = A.Compose([
    A.RandomCrop(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Split AOIs for train/val
train_aois = AOIS[:4]
val_aois = AOIS[4:]

train_ds = SpaceNet7ChangeDataset(DATA_ROOT, train_aois, transform=train_transform)
val_ds = SpaceNet7ChangeDataset(DATA_ROOT, val_aois, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)

print(f"Train: {len(train_ds)} pairs | Val: {len(val_ds)} pairs")


# ============================================================
# CELL 5: Siamese U-Net
# ============================================================

class SiameseEncoder(nn.Module):
    """Shared-weight encoder using a pretrained ResNet34 backbone."""
    def __init__(self):
        super().__init__()
        encoder = smp.encoders.get_encoder("resnet34", in_channels=3,
                                            depth=5, weights="imagenet")
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)


class DiffDecoder(nn.Module):
    """U-Net-style decoder that takes difference features."""
    def __init__(self, encoder_channels):
        super().__init__()
        # encoder_channels example for resnet34: [3, 64, 64, 128, 256, 512]
        chs = list(reversed(encoder_channels))  # [512, 256, 128, 64, 64, 3]

        self.up_blocks = nn.ModuleList()
        for i in range(len(chs) - 1):
            in_ch = chs[i] if i == 0 else chs[i] + chs[i]  # skip connection
            out_ch = chs[i + 1]
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))

        self.final = nn.Conv2d(chs[-1], 1, 1)

    def forward(self, diff_features):
        # diff_features: list from deepest to shallowest
        x = diff_features[0]
        for i, block in enumerate(self.up_blocks):
            if i > 0 and i < len(diff_features):
                # Resize if needed and concatenate skip
                skip = diff_features[i]
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                     align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = block(x)
        # Final resize to match input
        return self.final(x)


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for change detection.
    Shared encoder processes both images, absolute difference of
    feature maps is decoded to predict the change mask.
    """
    def __init__(self):
        super().__init__()
        self.encoder = SiameseEncoder()
        enc_channels = self.encoder.encoder.out_channels
        self.decoder = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=6,   # Concatenate both images
            classes=1,
            activation=None,
        )

    def forward(self, img1, img2):
        # Concatenate along channel dimension
        x = torch.cat([img1, img2], dim=1)  # (B, 6, H, W)
        return self.decoder(x)


# Simpler and more robust Siamese U-Net (concatenation-based)
class SiameseUNetV2(nn.Module):
    """
    Siamese U-Net V2: Two-stream encoder with feature difference + concat.
    Uses shared-weight ResNet34 encoder, computes |f1 - f2| at each level,
    then decodes with a standard U-Net decoder.
    """
    def __init__(self):
        super().__init__()
        # Shared encoder
        self.encoder = smp.encoders.get_encoder(
            "resnet34", in_channels=3, depth=5, weights="imagenet"
        )
        enc_channels = list(self.encoder.out_channels)  # e.g. [3,64,64,128,256,512]

        # Decoder takes difference features
        # Double channels because we concat |f1-f2| and f1+f2
        decoder_in = [c * 2 for c in enc_channels]

        self.decoder = nn.ModuleList()
        chs = list(reversed(decoder_in))
        for i in range(len(chs) - 1):
            in_c = chs[i] if i == 0 else chs[i] + out_c
            out_c = chs[i+1] // 2
            self.decoder.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ))
            self.decoder.append(nn.Upsample(scale_factor=2, mode='bilinear',
                                           align_corners=False))

        self.head = nn.Conv2d(out_c, 1, 1)

    def forward(self, img1, img2):
        f1 = self.encoder(img1)
        f2 = self.encoder(img2)

        # Compute difference and sum features
        diff_feats = []
        for a, b in zip(f1, f2):
            diff_feats.append(torch.cat([torch.abs(a - b), a + b], dim=1))

        # Decode from deepest
        x = diff_feats[-1]
        for i, (feat, layer) in enumerate(
            zip(reversed(diff_feats[:-1]),
                zip(self.decoder[::2], self.decoder[1::2]))):
            conv, up = layer
            x = conv(x)
            x = up(x)
            if x.shape[2:] != feat.shape[2:]:
                x = F.interpolate(x, size=feat.shape[2:], mode='bilinear',
                                 align_corners=False)
            x = torch.cat([x, feat], dim=1) if i < len(diff_feats)-2 else x

        x = self.head(x)
        return F.interpolate(x, size=img1.shape[2:], mode='bilinear',
                            align_corners=False)


# Use the simpler concat-based version
siamese_unet = SiameseUNet()
print(f"Siamese U-Net params: {sum(p.numel() for p in siamese_unet.parameters() if p.requires_grad):,}")


# ============================================================
# CELL 6: ChangeFormer (Transformer-based Change Detection)
# ============================================================

class PatchEmbed(nn.Module):
    """Image to patch embedding."""
    def __init__(self, in_channels=3, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                             stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, H, W


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block."""
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class ChangeFormer(nn.Module):
    """
    Simplified ChangeFormer for bi-temporal change detection.

    Architecture:
    1. Shared Siamese encoder (hierarchical transformer)
    2. Difference module
    3. MLP decoder head

    Based on: "A Transformer-Based Siamese Network for Change Detection"
    """
    def __init__(self, in_channels=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], depths=[2, 2, 2, 2],
                 img_size=256):
        super().__init__()
        self.stages = len(embed_dims)

        # Hierarchical encoder stages
        self.patch_embeds = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()

        for i in range(self.stages):
            in_ch = in_channels if i == 0 else embed_dims[i-1]
            patch_size = 4 if i == 0 else 2
            self.patch_embeds.append(PatchEmbed(in_ch, embed_dims[i], patch_size))

            blocks = nn.ModuleList([
                TransformerBlock(embed_dims[i], num_heads[i])
                for _ in range(depths[i])
            ])
            self.transformer_blocks.append(blocks)

        # Difference neck
        self.diff_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ) for dim in embed_dims
        ])

        # Decoder (FPN-like)
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for i in range(self.stages):
            self.lateral_convs.append(
                nn.Conv2d(embed_dims[i], 128, 1)
            )
            self.output_convs.append(nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ))

        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def _encode(self, x):
        """Hierarchical encoding of one image."""
        features = []
        for i in range(self.stages):
            x_patch, H, W = self.patch_embeds[i](x)
            for block in self.transformer_blocks[i]:
                x_patch = block(x_patch)

            B, N, C = x_patch.shape
            x = x_patch.transpose(1, 2).view(B, C, H, W)
            features.append(x)
        return features

    def forward(self, img1, img2):
        # Shared-weight encoding
        feats1 = self._encode(img1)
        feats2 = self._encode(img2)

        # Compute difference at each scale
        diff_feats = []
        for i, (f1, f2) in enumerate(zip(feats1, feats2)):
            diff = torch.abs(f1 - f2)
            diff = self.diff_convs[i](diff)
            diff_feats.append(diff)

        # FPN-style decoding
        # Start from deepest
        x = self.lateral_convs[-1](diff_feats[-1])
        x = self.output_convs[-1](x)

        for i in range(self.stages - 2, -1, -1):
            lateral = self.lateral_convs[i](diff_feats[i])
            x = F.interpolate(x, size=lateral.shape[2:], mode='bilinear',
                             align_corners=False)
            x = x + lateral
            x = self.output_convs[i](x)

        # Upsample to original resolution
        x = F.interpolate(x, size=img1.shape[2:], mode='bilinear',
                         align_corners=False)
        return self.head(x)


changeformer = ChangeFormer(img_size=IMG_SIZE)
print(f"ChangeFormer params: {sum(p.numel() for p in changeformer.parameters() if p.requires_grad):,}")


# ============================================================
# CELL 7: Loss & Metrics
# ============================================================

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        return 0.5 * bce_loss + 0.5 * dice_loss.mean()


class FocalLoss(nn.Module):
    """Focal loss — helps with class imbalance in change detection."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class CombinedChangeLoss(nn.Module):
    """Dice + Focal loss for change detection."""
    def __init__(self):
        super().__init__()
        self.dice_bce = DiceBCELoss()
        self.focal = FocalLoss()

    def forward(self, logits, targets):
        return 0.5 * self.dice_bce(logits, targets) + 0.5 * self.focal(logits, targets)


def compute_change_metrics(preds, targets, threshold=0.5):
    """Compute IoU, F1, precision, recall for change detection."""
    preds_bin = (torch.sigmoid(preds) > threshold).float()

    tp = (preds_bin * targets).sum()
    fp = (preds_bin * (1 - targets)).sum()
    fn = ((1 - preds_bin) * targets).sum()
    tn = ((1 - preds_bin) * (1 - targets)).sum()

    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)

    return {
        'iou': iou.item(),
        'f1': f1.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }


# ============================================================
# CELL 8: Training Loop
# ============================================================

def train_change_model(model, model_name, train_loader, val_loader,
                       num_epochs=30, lr=1e-4, device=DEVICE):
    model = model.to(device)
    criterion = CombinedChangeLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_iou = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_iou': [], 'val_f1': [],
        'val_precision': [], 'val_recall': []
    }

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for img1, img2, mask in tqdm(train_loader,
                                      desc=f"[{model_name}] Ep {epoch+1}/{num_epochs}"):
            img1 = img1.to(device)
            img2 = img2.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            output = model(img1, img2)
            loss = criterion(output, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        metrics_accum = {'iou': 0, 'f1': 0, 'precision': 0, 'recall': 0}
        n_batches = 0

        with torch.no_grad():
            for img1, img2, mask in val_loader:
                img1, img2, mask = img1.to(device), img2.to(device), mask.to(device)
                output = model(img1, img2)
                val_loss += criterion(output, mask).item()
                m = compute_change_metrics(output, mask)
                for k in metrics_accum:
                    metrics_accum[k] += m[k]
                n_batches += 1

        val_loss /= max(n_batches, 1)
        for k in metrics_accum:
            metrics_accum[k] /= max(n_batches, 1)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(metrics_accum['iou'])
        history['val_f1'].append(metrics_accum['f1'])
        history['val_precision'].append(metrics_accum['precision'])
        history['val_recall'].append(metrics_accum['recall'])

        print(f"  Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"IoU: {metrics_accum['iou']:.4f} | F1: {metrics_accum['f1']:.4f} | "
              f"P: {metrics_accum['precision']:.4f} | R: {metrics_accum['recall']:.4f}")

        if metrics_accum['iou'] > best_iou:
            best_iou = metrics_accum['iou']
            torch.save(model.state_dict(), f"/content/{model_name}_best.pth")
            print(f"  ✅ Saved {model_name} (IoU: {best_iou:.4f})")

    return model, history


# ============================================================
# CELL 9: Train Siamese U-Net
# ============================================================
print("=" * 60)
print("Training Siamese U-Net")
print("=" * 60)

siamese_unet, siamese_history = train_change_model(
    siamese_unet, "SiameseUNet",
    train_loader, val_loader,
    num_epochs=30, lr=1e-4
)


# ============================================================
# CELL 10: Train ChangeFormer
# ============================================================
print("=" * 60)
print("Training ChangeFormer")
print("=" * 60)

changeformer, cf_history = train_change_model(
    changeformer, "ChangeFormer",
    train_loader, val_loader,
    num_epochs=30, lr=3e-5  # Lower LR for transformer
)


# ============================================================
# CELL 11: Plot Training Curves
# ============================================================

def plot_change_curves(histories, names):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for hist, name in zip(histories, names):
        axes[0,0].plot(hist['train_loss'], label=f'{name} Train')
        axes[0,0].plot(hist['val_loss'], '--', label=f'{name} Val')
    axes[0,0].set_title('Loss'); axes[0,0].legend(); axes[0,0].grid(True)

    for hist, name in zip(histories, names):
        axes[0,1].plot(hist['val_iou'], label=name)
    axes[0,1].set_title('Val IoU'); axes[0,1].legend(); axes[0,1].grid(True)

    for hist, name in zip(histories, names):
        axes[1,0].plot(hist['val_f1'], label=name)
    axes[1,0].set_title('Val F1'); axes[1,0].legend(); axes[1,0].grid(True)

    for hist, name in zip(histories, names):
        axes[1,1].plot(hist['val_precision'], label=f'{name} Prec')
        axes[1,1].plot(hist['val_recall'], '--', label=f'{name} Rec')
    axes[1,1].set_title('Precision & Recall'); axes[1,1].legend(); axes[1,1].grid(True)

    plt.suptitle('Model 4 — Change Detection Training', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('/content/model4_training_curves.png', dpi=150)
    plt.show()

plot_change_curves(
    [siamese_history, cf_history],
    ['Siamese U-Net', 'ChangeFormer']
)


# ============================================================
# CELL 12: Visualize Change Detection Predictions
# ============================================================

def visualize_change_predictions(model, name, dataloader, n=4):
    model.eval()
    img1, img2, masks = next(iter(dataloader))
    img1, img2 = img1.to(DEVICE), img2.to(DEVICE)

    with torch.no_grad():
        preds = torch.sigmoid(model(img1, img2)) > 0.5

    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    cols = ['Image T1', 'Image T2', 'Ground Truth', f'{name} Prediction']

    for i in range(min(n, img1.shape[0])):
        def denorm(t):
            t = t.cpu().numpy().transpose(1, 2, 0)
            return (t - t.min()) / (t.max() - t.min() + 1e-8)

        axes[i, 0].imshow(denorm(img1[i]))
        axes[i, 1].imshow(denorm(img2[i]))
        axes[i, 2].imshow(masks[i, 0].cpu(), cmap='RdYlGn_r')
        axes[i, 3].imshow(preds[i, 0].cpu(), cmap='RdYlGn_r')

        if i == 0:
            for j, c in enumerate(cols):
                axes[i, j].set_title(c, fontsize=12)
        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'/content/{name}_change_predictions.png', dpi=150)
    plt.show()

visualize_change_predictions(siamese_unet, "Siamese U-Net", val_loader)
visualize_change_predictions(changeformer, "ChangeFormer", val_loader)


# ============================================================
# CELL 13: Final Summary
# ============================================================

print("\n" + "=" * 60)
print("MODEL 4 — TEMPORAL CHANGE DETECTION RESULTS")
print("=" * 60)
print(f"Siamese U-Net  — Best IoU: {max(siamese_history['val_iou']):.4f} | "
      f"Best F1: {max(siamese_history['val_f1']):.4f}")
print(f"ChangeFormer   — Best IoU: {max(cf_history['val_iou']):.4f} | "
      f"Best F1: {max(cf_history['val_f1']):.4f}")

# Save to Drive (optional)
# import shutil
# for f in ['SiameseUNet_best.pth', 'ChangeFormer_best.pth']:
#     shutil.copy(f'/content/{f}', f'/content/drive/MyDrive/{f}')

print("\nSaved models:")
print("  /content/SiameseUNet_best.pth")
print("  /content/ChangeFormer_best.pth")
