"""Inference-only definitions from NikhilR5577/AstraGuard-Models; see vendor/SOURCE.json. No pretrained downloads."""
import torch
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp

class SiameseEncoder(nn.Module):
    """Shared-weight encoder using a pretrained ResNet34 backbone."""
    def __init__(self):
        super().__init__()
        encoder = smp.encoders.get_encoder("resnet34", in_channels=3,
                                            depth=5, weights=None)
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)

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