import torch
import torch.nn as nn
import sys
import os.path as osp
import numpy as np
from yacs.config import CfgNode as CN
this_dir = osp.dirname(__file__)
sys.path.insert(0, this_dir)
import tgs
from network.pointnet import PointNetEncoder

hort_cfg = CN()
hort_cfg.image_tokenizer_cls = "tgs.models.tokenizers.image.DINOV2SingleImageTokenizer"
hort_cfg.image_tokenizer = CN()
hort_cfg.image_tokenizer.pretrained_model_name_or_path = "facebook/dinov2-large"
hort_cfg.image_tokenizer.width = 224
hort_cfg.image_tokenizer.height = 224
hort_cfg.image_tokenizer.modulation = False
hort_cfg.image_tokenizer.modulation_zero_init = True
hort_cfg.image_tokenizer.modulation_cond_dim = 1024
hort_cfg.image_tokenizer.freeze_backbone_params = False
hort_cfg.image_tokenizer.enable_memory_efficient_attention = False
hort_cfg.image_tokenizer.enable_gradient_checkpointing = False

hort_cfg.tokenizer_cls = "tgs.models.tokenizers.point.PointLearnablePositionalEmbedding"
hort_cfg.tokenizer = CN()
hort_cfg.tokenizer.num_pcl = 2049
hort_cfg.tokenizer.num_channels = 512

hort_cfg.backbone_cls = "tgs.models.transformers.Transformer1D"
hort_cfg.backbone = CN()
hort_cfg.backbone.in_channels = 512
hort_cfg.backbone.num_attention_heads = 8
hort_cfg.backbone.attention_head_dim = 64
hort_cfg.backbone.num_layers = 10
hort_cfg.backbone.cross_attention_dim = 1024
hort_cfg.backbone.norm_type = "layer_norm"
hort_cfg.backbone.enable_memory_efficient_attention = False
hort_cfg.backbone.gradient_checkpointing = False

hort_cfg.post_processor_cls = "tgs.models.networks.PointOutLayer"
hort_cfg.post_processor = CN()
hort_cfg.post_processor.in_channels = 512
hort_cfg.post_processor.out_channels = 3

hort_cfg.pointcloud_upsampler_cls = "tgs.models.snowflake.model_spdpp.SnowflakeModelSPDPP"
hort_cfg.pointcloud_upsampler = CN()
hort_cfg.pointcloud_upsampler.input_channels = 1024
hort_cfg.pointcloud_upsampler.dim_feat = 128
hort_cfg.pointcloud_upsampler.num_p0 = 2048
hort_cfg.pointcloud_upsampler.radius = 1
hort_cfg.pointcloud_upsampler.bounding = True
hort_cfg.pointcloud_upsampler.use_fps = True
hort_cfg.pointcloud_upsampler.up_factors = [2, 4]
hort_cfg.pointcloud_upsampler.token_type = "image_token"


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.image_tokenizer = tgs.find(hort_cfg.image_tokenizer_cls)(hort_cfg.image_tokenizer)
        self.pointnet = PointNetEncoder(67, 1024)
        self.tokenizer = tgs.find(hort_cfg.tokenizer_cls)(hort_cfg.tokenizer)
        self.backbone = tgs.find(hort_cfg.backbone_cls)(hort_cfg.backbone)
        self.post_processor = tgs.find(hort_cfg.post_processor_cls)(hort_cfg.post_processor)
        self.post_processor_trans = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        self.pointcloud_upsampler = tgs.find(hort_cfg.pointcloud_upsampler_cls)(hort_cfg.pointcloud_upsampler)
    
    def forward(self, input_img, metas):
        with torch.no_grad():
            batch_size = input_img.shape[0]

            encoder_hidden_states = self.image_tokenizer(input_img, None) # B * C * Nt
            encoder_hidden_states = encoder_hidden_states.transpose(2, 1) # B * Nt * C

            palm_norm_hand_verts_3d = metas['right_hand_verts_3d'] - metas['right_hand_palm'].unsqueeze(1)
            point_idx = torch.arange(778).view(1, 778, 1).expand(batch_size, -1, -1).to(input_img.device) / 778.
            palm_norm_hand_verts_3d = torch.cat([palm_norm_hand_verts_3d, point_idx], -1)
            tip_norm_hand_verts_3d = (metas['right_hand_verts_3d'].unsqueeze(2) - metas['right_hand_joints_3d'].unsqueeze(1)).reshape((batch_size, 778, -1))
            norm_hand_verts_3d = torch.cat([palm_norm_hand_verts_3d, tip_norm_hand_verts_3d], -1)
            hand_feats = self.pointnet(norm_hand_verts_3d)

            tokens = self.tokenizer(batch_size)
            tokens = self.backbone(tokens, torch.cat([encoder_hidden_states, hand_feats.unsqueeze(1)], 1), modulation_cond=None)
            tokens = self.tokenizer.detokenize(tokens)

            pointclouds = self.post_processor(tokens[:, :2048, :])
            pred_obj_trans = self.post_processor_trans(tokens[:, -1, :])

            upsampling_input = {
                "input_image_tokens": encoder_hidden_states.permute(0, 2, 1),
                "intrinsic_cond": metas['cam_intr'],
                "points": pointclouds,
                "hand_points": metas["right_hand_verts_3d"],
                "trans": pred_obj_trans + metas['right_hand_palm'],
                "scale": 0.3
            }
            up_results = self.pointcloud_upsampler(upsampling_input)
            pointclouds_up = up_results[-1]

            pc_results = {}
            pc_results['pointclouds'] = pointclouds
            pc_results['objtrans'] = pred_obj_trans
            pc_results['handpalm'] = metas['right_hand_palm']
            pc_results['pointclouds_up'] = pointclouds_up

            return pc_results

def load_hort(ckpt_path):
    hort_model = model()
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))["network"]
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    hort_model.load_state_dict(ckpt)
    return hort_model
