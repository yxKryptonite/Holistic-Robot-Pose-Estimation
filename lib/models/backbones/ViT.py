import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ViTBackbone(nn.Module):
    """
    Wrapper around timm ViT that:
      - takes RGB images (B, 3, H, W)
      - returns:
          out: (B, num_joints * depth_dim, H/4, W/4)  -> for integral layer
          xf:  (B, C_feat)                            -> global features for MLPs
    """

    def __init__(self, backbone_name, num_joints, depth_dim, image_size=224):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_joints = num_joints
        self.depth_dim = depth_dim
        self.image_size = int(image_size)

        # timm ViT, last feature map only
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=(-1,)
        )

        # channels of last feature map (e.g. 384 for vit_small)
        feat_channels = self.backbone.feature_info.channels()[-1]

        # Small conv head to go from ViT feature map -> heatmaps
        self.conv1 = nn.Conv2d(
            feat_channels, 256, kernel_size=3, padding=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(
            256, num_joints * depth_dim, kernel_size=1, bias=True
        )

        # Global pool for regression / rotation branches
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # feats[-1]: (B, C, H_feat, W_feat), e.g. (B, C, 7, 7) for 224x224 with patch 32
        feats = self.backbone(x)
        feat = feats[-1]
        B, C, H, W = feat.shape

        # 1) Build heatmaps at spatial size image_size/4 (e.g. 56x56)
        xh = self.conv1(feat)
        xh = self.relu(xh)
        target_hw = self.image_size // 4  # 224 -> 56
        xh = F.interpolate(
            xh,
            size=(target_hw, target_hw),
            mode="bilinear",
            align_corners=False,
        )
        out = self.conv_out(xh)  # (B, num_joints * depth_dim, 56, 56)

        # 2) Global feature for regression heads
        xf = self.global_pool(feat).view(B, C)  # (B, C_feat)

        return out, xf


def get_vit_backbone(backbone_name, num_joints, depth_dim, image_size=224):
    model = ViTBackbone(
        backbone_name=backbone_name,
        num_joints=num_joints,
        depth_dim=depth_dim,
        image_size=image_size,
    )
    feat_channels = model.backbone.feature_info.channels()[-1]
    print(f"Using timm pretrained weights for ViT backbone: {backbone_name}")
    return model, feat_channels

