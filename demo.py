from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
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


def main():
    parser = argparse.ArgumentParser(description='HORT demo code')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'], help='List of file extensions to consider')

    args = parser.parse_args()

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

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    
    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(img_path)
        img_pil = Image.fromarray(img_cv2[..., ::-1])

        pred_obj = sam_model.predict([img_pil], ["manipulated object"])
        bbox_obj = pred_obj[0]["boxes"][0].reshape((-1, 2))

        detections = hand_detector(img_cv2, conf=0.3, verbose=False)[0]

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
            
        dataset = ViTDetDataset(wilor_model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
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
            
            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                
                verts = out['pred_vertices'][n].detach().cpu().numpy()
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

                # Save all results to disk
                mano_faces = np.load(os.path.join('.', 'mano_data', 'closed_fmano.npy'))
                hand_mesh = trimesh.Trimesh(vertices=verts + cam_t, faces=mano_faces)
                hand_mesh.export(os.path.join(args.out_folder, f'{img_fn}.obj'))

                with open(os.path.join(args.out_folder, f'{img_fn}.json'), 'w') as f:
                    output_dict = dict()
                    output_dict['cam_extr'] = np.eye(3, dtype=np.float32).tolist()
                    output_dict['cam_intr'] = cam_intr.tolist()
                    output_dict['pointclouds_up'] = pointclouds_up.tolist()
                    output_dict['objtrans'] = objtrans.tolist()
                    output_dict['handpalm'] = (palm + cam_t).tolist()
                    json.dump(output_dict, f)

                misc_args = dict(mesh_base_color=LIGHT_PURPLE, point_base_color=STEEL_BLUE, scene_bg_color=(1, 1, 1), focal_length=scaled_focal_length)
                cam_view = renderer.render_rgba(verts, pointclouds_up + palm + objtrans, cam_t=cam_t, render_res=(224, 224), is_right=True, **misc_args)

                # Overlay image
                input_img = crop_img_cv2.astype(np.float32)[:,:,::-1] / 255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

                cv2.imwrite(os.path.join(args.out_folder, f"{img_fn}.jpg"), 255 * input_img_overlay[:, :, ::-1])

if __name__ == '__main__':
    main()
