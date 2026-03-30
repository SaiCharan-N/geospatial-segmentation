import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class RoadModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ✅ SAME backbone used during training
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2",
            pretrained=False,   # IMPORTANT: weights come from .pth
            num_classes=0
        )

        # allow flexible image sizes
        self.backbone.patch_embed.strict_img_size = False

        self.embed_dim = self.backbone.num_features

        # ✅ EXACT SAME decoder as training
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embedding
        x = self.backbone.patch_embed(x)

        # Add CLS token
        cls = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)

        # ✅ Position embedding interpolation (CRITICAL)
        pos = self.backbone.pos_embed
        cls_pos, patch_pos = pos[:, :1], pos[:, 1:]

        n = x.shape[1] - 1
        orig = int(patch_pos.shape[1] ** 0.5)
        new = int(n ** 0.5)

        patch_pos = patch_pos.reshape(1, orig, orig, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, (new, new),
            mode='bilinear',
            align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new * new, -1)

        pos = torch.cat((cls_pos, patch_pos), dim=1)
        x = x + pos

        # Transformer blocks
        for blk in self.backbone.blocks:
            x = blk(x)

        x = self.backbone.norm(x)

        # Remove CLS token
        x = x[:, 1:, :]

        h = w = int(x.shape[1] ** 0.5)

        # Reshape to feature map
        x = x.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)

        # Decoder
        x = self.decoder(x)

        # Upsample to original size
        x = F.interpolate(
            x,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        return x


# ✅ Loader function (used by your pipeline)
def get_road_model():
    return RoadModel()