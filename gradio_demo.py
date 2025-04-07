import os 
import sys 
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"

import gradio as gr
#import spaces
import cv2 
import numpy as np 
import torch 
from ultralytics import YOLO
from pathlib import Path
import argparse
import json
import trimesh
from torchvision import transforms
from typing import Dict, Optional
from PIL import Image, ImageDraw
from lang_sam import LangSAM

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from hort.models import load_hort
from hort.utils.renderer import Renderer, cam_crop_to_new
from hort.utils.img_utils import process_bbox, generate_patch_image, PerspectiveCamera
from ultralytics import YOLO
LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)
STEEL_BLUE=(0.2745098, 0.5098039, 0.7058824)


def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    # Compute intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection = inter_width * inter_height
    # Compute areas of each box
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Compute union
    union = area_box1 + area_box2 - intersection
    # Compute IoU
    return intersection / union if union > 0 else 0.0


# Download and load checkpoints
wilor_model, wilor_model_cfg = load_wilor(checkpoint_path = './pretrained_models/wilor_final.ckpt' , cfg_path= './pretrained_models/model_config.yaml')
hand_detector = YOLO('./pretrained_models/detector.pt')
# Setup the renderer
renderer = Renderer(wilor_model_cfg, faces=wilor_model.mano.faces)
# Setup the SAM model
sam_model = LangSAM(sam_type="sam2.1_hiera_large")
# Setup the HORT model
hort_model = load_hort("./pretrained_models/hort_final.pth.tar")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
wilor_model = wilor_model.to(device)
hand_detector = hand_detector.to(device)
hort_model = hort_model.to(device)
wilor_model.eval()
hort_model.eval()

image_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#@spaces.GPU()
def run_model(image, conf, IoU_threshold=0.5):
    img_cv2 = image[..., ::-1]
    img_pil = Image.fromarray(image)

    pred_obj = sam_model.predict([img_pil], ["manipulated object"])
    bbox_obj = pred_obj[0]["boxes"][0].reshape((-1, 2))

    detections = hand_detector(img_cv2, conf=conf, verbose=False, iou=IoU_threshold)[0]
    
    bboxes = []
    is_right = []
    for det in detections: 
        Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(Bbox[:4].tolist())

    if len(bboxes) == 0:
        print("no hands in this image")
    elif len(bboxes) == 1:
        bbox_hand = np.array(bboxes[0]).reshape((-1, 2))
    elif len(bboxes) > 1:
        hand_idx = None
        max_iou = -10.
        for cur_idx, cur_bbox in enumerate(bboxes):
            cur_iou = calculate_iou(cur_bbox, bbox_obj.reshape(-1).tolist())
            if cur_iou >= max_iou:
                hand_idx = cur_idx
                max_iou = cur_iou
        bbox_hand = np.array(bboxes[hand_idx]).reshape((-1, 2))
        bboxes = [bboxes[hand_idx]]
        is_right = [is_right[hand_idx]]

    tl = np.min(np.concatenate([bbox_obj, bbox_hand], axis=0), axis=0)
    br = np.max(np.concatenate([bbox_obj, bbox_hand], axis=0), axis=0)
    box_size = br - tl
    bbox = np.concatenate([tl - 10, box_size + 20], axis=0)
    ho_bbox = process_bbox(bbox)
          
    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    if not right:
        new_x1 = img_cv2.shape[1] - boxes[0][2]
        new_x2 = img_cv2.shape[1] - boxes[0][0]
        boxes[0][0] = new_x1
        boxes[0][2] = new_x2
        ho_bbox[0] = img_cv2.shape[1] - (ho_bbox[0] + ho_bbox[2])
        img_cv2 = cv2.flip(img_cv2, 1)
        right[0] = 1.
    crop_img_cv2, _ = generate_patch_image(img_cv2, ho_bbox, (224, 224), 0, 1.0, 0)
        
    dataset = ViTDetDataset(wilor_model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    for batch in dataloader: 
        batch = recursive_to(batch, device)
    
        with torch.no_grad():
            out = wilor_model(batch) 
            
        pred_cam      = out['pred_cam']
        box_center    = batch["box_center"].float()
        box_size      = batch["box_size"].float()
        img_size      = batch["img_size"].float()
        scaled_focal_length = wilor_model_cfg.EXTRA.FOCAL_LENGTH / wilor_model_cfg.MODEL.IMAGE_SIZE * 224
        pred_cam_t_full = cam_crop_to_new(pred_cam, box_center, box_size, img_size, torch.from_numpy(np.array(ho_bbox, dtype=np.float32))[None, :].to(img_size.device), scaled_focal_length).detach().cpu().numpy()
        
        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            verts  = out['pred_vertices'][n].detach().cpu().numpy()
            joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
            
            is_right = batch['right'][n].cpu().numpy()
            palm = (verts[95] + verts[22]) / 2
            cam_t = pred_cam_t_full[n]

            img_input = image_transform(crop_img_cv2[:, :, ::-1]).unsqueeze(0).cuda()
            camera = PerspectiveCamera(5000 / 256 * 224, 5000 / 256 * 224, 112, 112)
            cam_intr = camera.intrinsics

            metas = dict()
            metas["right_hand_verts_3d"] = torch.from_numpy((verts + cam_t)[None]).cuda()
            metas["right_hand_joints_3d"] = torch.from_numpy((joints + cam_t)[None]).cuda()
            metas["right_hand_palm"] = torch.from_numpy((palm + cam_t)[None]).cuda()
            metas["cam_intr"] = torch.from_numpy(cam_intr[None]).cuda()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                pc_results = hort_model(img_input, metas)
            objtrans = pc_results["objtrans"][0].detach().cpu().numpy()
            pointclouds_up = pc_results["pointclouds_up"][0].detach().cpu().numpy() * 0.3
            
            reconstructions = {'verts': verts, 'palm': palm, 'objtrans': objtrans, 'objpcs': pointclouds_up, 'cam_t': cam_t, 'right': is_right, 'img_size': 224, 'focal': scaled_focal_length}
        
            camera_translation = cam_t.copy()
            hand_mesh = renderer.mesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right)
            obj_pcd = trimesh.PointCloud(reconstructions['objpcs'] + reconstructions['palm'] + reconstructions['objtrans'] + camera_translation, colors=[70, 130, 180, 255])
            scene = trimesh.Scene([hand_mesh, obj_pcd])
            scene_path = "out_demo/test.glb"
            scene.export(scene_path)

        return crop_img_cv2[..., ::-1].astype(np.float32) / 255.0, len(detections), reconstructions, scene_path


def render_reconstruction(image, conf, IoU_threshold=0.3): 
    input_img, num_dets, reconstructions, scene_path = run_model(image, conf, IoU_threshold=0.5)
    misc_args = dict(mesh_base_color=LIGHT_PURPLE, point_base_color=STEEL_BLUE, scene_bg_color=(1, 1, 1), focal_length=reconstructions['focal'])
    cam_view = renderer.render_rgba(reconstructions['verts'], reconstructions['objpcs'] + reconstructions['palm'] + reconstructions['objtrans'], cam_t=reconstructions['cam_t'], render_res=(224, 224), is_right=True, **misc_args)

    # Overlay image
    input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
    input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

    return input_img_overlay, f'{num_dets} hands detected', scene_path


header = ('''
<div class="embed_hidden" style="text-align: center;">
    <h1> <b>HORT</b>: Monocular Hand-held Objects Reconstruction with Transformers</h1>
    <h3>
        <a href="https://zerchen.github.io/" target="_blank" rel="noopener noreferrer">Zerui Chen</a><sup>1</sup>,
        <a href="https://rolpotamias.github.io" target="_blank" rel="noopener noreferrer">Rolandos Alexandros Potamias</a><sup>2</sup>,
        <br>
        <a href="https://cshizhe.github.io/" target="_blank" rel="noopener noreferrer">Shizhe Chen</a><sup>1</sup>,
        <a href="https://cordeliaschmid.github.io/" target="_blank" rel="noopener noreferrer">Cordelia Schmid</a><sup>1</sup>
    </h3>
    <h3>
        <sup>1</sup>Inria, Ecole normale supérieure, CNRS, PSL Research University;
        <sup>2</sup>Imperial College London
    </h3>
</div>
<div style="display:flex; gap: 0.3rem; justify-content: center; align-items: center;" align="center">
<a href='https://arxiv.org/abs/2503.21313'><img src='https://img.shields.io/badge/Arxiv-2503.21313-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
<a href='https://arxiv.org/pdf/2503.21313'><img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a> 
<a href='https://zerchen.github.io/projects/hort.html'><img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
<a href='https://github.com/zerchen/hort'><img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a> 
''')


theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)
with gr.Blocks(theme=theme, title="HORT: Monocular Hand-held Objects Reconstruction with Transformers", css=".gradio-container") as demo:

    gr.Markdown(header)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input image", type="numpy")
            threshold = gr.Slider(value=0.3, minimum=0.05, maximum=0.95, step=0.05, label='Detection Confidence Threshold')
            submit = gr.Button("Submit", variant="primary")
            example_images = gr.Examples([
                ['./demo_img/test1.png'], 
                ['./demo_img/test2.png'], 
                ['./demo_img/test3.jpg'], 
                ['./demo_img/test4.jpg'],
                ['./demo_img/test5.jpeg'],
                ['./demo_img/test6.jpg'],
                ['./demo_img/test7.jpg'],
                ['./demo_img/test8.jpeg']
                ], 
                inputs=input_image)
        
        with gr.Column():
            reconstruction = gr.Image(label="Reconstructions", type="numpy")
            output_meshes = gr.Model3D(height=300, zoom_speed=0.5, pan_speed=0.5)
            hands_detected = gr.Textbox(label="Hands Detected")
    
        submit.click(fn=render_reconstruction, inputs=[input_image, threshold], outputs=[reconstruction, hands_detected, output_meshes])

demo.launch()