import torch
import torch.nn as nn
import torch.nn.functional as F

from tgs.utils.base import BaseModule
from tgs.utils.typing import *
from dataclasses import dataclass, field

from pytorch3d.renderer import (
    AlphaCompositor,
    NormWeightedCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

from .utils import fps_subsample
from einops import rearrange

from .utils import MLP_CONV
from .SPD import SPD
from .SPD_crossattn import SPD_crossattn
from .SPD_pp import SPD_pp

SPD_BLOCK = {
    'SPD': SPD,
    'SPD_crossattn': SPD_crossattn,
    'SPD_PP': SPD_pp,
}


def homoify(points):
    """
    Convert a batch of points to homogeneous coordinates.
    Args:
        points: e.g. (B, N, 3) or (N, 3)
    Returns:
        homoified points: e.g., (B, N, 4)
    """
    points_dim = points.shape[:-1] + (1,)
    ones = points.new_ones(points_dim)

    return torch.cat([points, ones], dim=-1)


def dehomoify(points):
    """
    Convert a batch of homogeneous points to cartesian coordinates.
    Args:
        homogeneous points: (B, N, 4/3) or (N, 4/3)
    Returns:
        cartesian points: (B, N, 3/2)
    """
    return points[..., :-1] / points[..., -1:]


def mask_generation(points: Float[Tensor, "B Np 3"],
                      intrinsics: Float[Tensor, "B 3 3"],
                      input_img: Float[Tensor, "B C H W"],
                      raster_point_radius: float = 0.01,  # point size
                      raster_points_per_pixel: int = 1,  # a single point per pixel, for now
                      bin_size: int = 0):
    """
    points: (B, Np, 3)
    """
    B, C, H, W = input_img.shape
    device = intrinsics.device

    cam_R = torch.eye(3).to(device).unsqueeze(0).repeat(B, 1, 1)
    cam_t = torch.zeros(3).to(device).unsqueeze(0).repeat(B, 1)

    raster_settings = PointsRasterizationSettings(image_size=(H, W), radius=raster_point_radius, points_per_pixel=raster_points_per_pixel, bin_size=bin_size)

    image_size = torch.as_tensor([H, W]).view(1, 2).expand(B, -1).to(device)
    cameras = cameras_from_opencv_projection(cam_R, cam_t, intrinsics, image_size)

    rasterize = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterize(Pointclouds(points))

    fragments_idx: Tensor = fragments.idx.long()
    mask = (fragments_idx[..., 0] > -1)

    return mask.float()


def points_projection(points: Float[Tensor, "B Np 3"],
                      intrinsics: Float[Tensor, "B 3 3"],
                      local_features: Float[Tensor, "B C H W"],
                      raster_point_radius: float = 0.0075,  # point size
                      raster_points_per_pixel: int = 1,  # a single point per pixel, for now
                      bin_size: int = 0):
    """
    points: (B, Np, 3)
    """
    B, C, H, W = local_features.shape
    device = local_features.device
    cam_R = torch.eye(3).to(device).unsqueeze(0).repeat(B, 1, 1)
    cam_t = torch.zeros(3).to(device).unsqueeze(0).repeat(B, 1)

    raster_settings = PointsRasterizationSettings(image_size=(H, W), radius=raster_point_radius, points_per_pixel=raster_points_per_pixel, bin_size=bin_size)
    Np = points.shape[1]
    R = raster_settings.points_per_pixel
    image_size = torch.as_tensor([H, W]).view(1, 2).expand(B, -1).to(device)
    cameras = cameras_from_opencv_projection(cam_R, cam_t, intrinsics, image_size)
    rasterize = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterize(Pointclouds(points))
    fragments_idx: Tensor = fragments.idx.long()
    visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
    points_to_visible_pixels = fragments_idx[visible_pixels]
    # Reshape local features to (B, H, W, R, C)
    local_features = local_features.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R, -1)  # (B, H, W, R, C)
    # Get local features corresponding to visible points
    local_features_proj = torch.zeros(B * Np, C, device=device)
    local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]
    local_features_proj = local_features_proj.reshape(B, Np, C)
    return local_features_proj


def points_projection_v2(input_xyz_points, cam_intr, feature_maps):
    input_points = input_xyz_points.clone()
    batch_size = input_points.shape[0]
    xyz = input_points[:, :, :3]
    homo_xyz = homoify(xyz)
    homo_xyz_2d = torch.matmul(cam_intr, homo_xyz.transpose(1, 2)).transpose(1, 2)
    xyz_2d = (homo_xyz_2d[:, :, :2] / homo_xyz_2d[:, :, [2]]).unsqueeze(2)
    uv_2d = xyz_2d / 224 * 2 - 1
    sample_feat = torch.nn.functional.grid_sample(feature_maps, uv_2d, align_corners=True)[:, :, :, 0].transpose(1, 2)
    uv_2d = uv_2d.squeeze(2).reshape((-1, 2))
    validity = (uv_2d[:, 0] >= -1.0) & (uv_2d[:, 0] <= 1.0) & (uv_2d[:, 1] >= -1.0) & (uv_2d[:, 1] <= 1.0)
    validity = validity.unsqueeze(1)

    return sample_feat


class Decoder(nn.Module):
    def __init__(self, input_channels=1152, dim_feat=512, num_p0=512,
                 radius=1, bounding=True, up_factors=None,
                 SPD_type='SPD',
                 token_type='image_token'
                 ):
        super(Decoder, self).__init__()
        # self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_p0)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = up_factors
        uppers = []
        self.num_p0 = num_p0
        self.mlp_feat_cond = MLP_CONV(in_channel=input_channels,
                                      layer_dims=[dim_feat*2, dim_feat])

        for i, factor in enumerate(up_factors):
            uppers.append(
                SPD_BLOCK[SPD_type](dim_feat=dim_feat, up_factor=factor,
                                    i=i, bounding=bounding, radius=radius))
        self.uppers = nn.ModuleList(uppers)
        self.token_type = token_type

    def calculate_pcl_token(self, pcl_token, up_factor):
        up_token =  F.interpolate(pcl_token, scale_factor=up_factor, mode='nearest')
        return up_token

    def calculate_image_token(self, pcd, hand_pcd, input_image_tokens, batch):
        """
        Args:
        """
        batch_size = input_image_tokens.shape[0]
        h_cond, w_cond = 224, 224
        input_image_tokens = input_image_tokens.permute(0, 2, 1)
        local_features = input_image_tokens[:, 1:].reshape(batch_size, h_cond // 14, w_cond // 14, -1).permute(0, 3, 1, 2)
        # local_features = F.interpolate(local_features, size=(h_cond, w_cond), mode='bilinear', align_corners=False)
        local_features_proj = points_projection_v2(pcd * batch['scale'] + batch['trans'].unsqueeze(1), batch['intrinsic_cond'], local_features)
        local_features_proj = local_features_proj.permute(0, 2, 1).contiguous()

        local_features_proj_hand = points_projection_v2(hand_pcd, batch['intrinsic_cond'], local_features)
        local_features_proj_hand = local_features_proj_hand.permute(0, 2, 1).contiguous()
 
        return local_features_proj, local_features_proj_hand

    def forward(self, x):
        """
        Args:
            points: Tensor, (b, num_p0, 3)
            feat_cond: Tensor, (b, dim_feat) dinov2: 325x768
            # partial_coarse: Tensor, (b, n_coarse, 3)
        """
        points = x['points']
        hand_points = x['hand_points']
        if self.token_type == 'pcl_token':
            feat_cond = x['pcl_token']
        elif self.token_type == 'image_token':
            feat_cond = x['input_image_tokens']
        feat_cond = self.mlp_feat_cond(feat_cond)
        arr_pcd = []
        feat_prev = None

        pcd = torch.permute(points, (0, 2, 1)).contiguous()
        pcl_up_scale = 1
        hand_pcd = torch.permute((hand_points - x['trans'].unsqueeze(1)) / x['scale'], (0, 2, 1)).contiguous()
        for upper in self.uppers:
            if self.token_type == 'pcl_token':
                up_cond = self.calculate_pcl_token(
                    feat_cond, pcl_up_scale)
                pcl_up_scale *= upper.up_factor
            elif self.token_type == 'image_token':
                up_cond, up_cond_hand = self.calculate_image_token(points, hand_points, feat_cond, x)
            pcd, feat_prev = upper(pcd, hand_pcd, up_cond_hand, up_cond, feat_prev)
            points = torch.permute(pcd, (0, 2, 1)).contiguous()
            arr_pcd.append(points)
        return arr_pcd


class SnowflakeModelSPDPP(BaseModule):
    """
    apply PC^2 / PCL token to decoder
    """
    @dataclass
    class Config(BaseModule.Config):
        input_channels: int = 1152
        dim_feat: int = 128
        num_p0: int = 512
        radius: float = 1
        bounding: bool = True
        use_fps: bool = True
        up_factors: List[int] = field(default_factory=lambda: [2, 2])
        image_full_token_cond: bool = False
        SPD_type: str = 'SPD_PP'
        token_type: str = 'pcl_token'
    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.decoder = Decoder(input_channels=self.cfg.input_channels,
                               dim_feat=self.cfg.dim_feat, num_p0=self.cfg.num_p0,
                               radius=self.cfg.radius, up_factors=self.cfg.up_factors, bounding=self.cfg.bounding,
                               SPD_type=self.cfg.SPD_type,
                               token_type=self.cfg.token_type
                               )

    def forward(self, x):
        results = self.decoder(x)
        return results
