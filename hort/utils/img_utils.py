import numpy as np
import cv2


def process_bbox(bbox, factor=1.25):
    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = 1.
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * factor
    bbox[3] = h * factor
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox


def generate_patch_image(cvimg, bbox, input_shape, do_flip, scale, rot):
    """
    @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                  generate the patch image from the bounding box and other parameters.
    ---------
    @param: input image, bbox(x1, y1, w, h), dest image shape, do_flip, scale factor, rotation degrees.
    -------
    @Returns: processed image, affine_transform matrix to get the processed image.
    -------
    """

    img = cvimg.copy()
    img_height, img_width, _ = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale, rot, inv=False)
    img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)
    new_trans = np.zeros((3, 3), dtype=np.float32)
    new_trans[:2, :] = trans
    new_trans[2, 2] = 1

    return img_patch, new_trans


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    """
    @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                  get affine transform matrix
    ---------
    @param: image center, original image size, desired image size, scale factor, rotation degree, whether to get inverse transformation.
    -------
    @Returns: affine transformation matrix
    -------
    """

    def rotate_2d(pt_2d, rot_rad):
        x = pt_2d[0]
        y = pt_2d[1]
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        xx = x * cs - y * sn
        yy = x * sn + y * cs
        return np.array([xx, yy], dtype=np.float32)

    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


class PerspectiveCamera:
    def __init__(self, fx, fy, cx, cy, R=np.eye(3), t=np.zeros(3)):
        self.K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float32)

        self.R = np.array(R, dtype=np.float32).copy()
        assert self.R.shape == (3, 3)

        self.t = np.array(t, dtype=np.float32).copy()
        assert self.t.size == 3
        self.t = self.t.reshape(3, 1)

    def update_virtual_camera_after_crop(self, bbox, option='same'):
        left, upper, width, height = bbox
        new_img_center = np.array([left + width / 2, upper + height / 2, 1], dtype=np.float32).reshape(3, 1)
        new_cam_center = np.linalg.inv(self.K[:3, :3]).dot(new_img_center)
        self.K[0, 2], self.K[1, 2] = width / 2, height / 2

        x, y, z = new_cam_center[0], new_cam_center[1], new_cam_center[2]
        sin_theta = -y / np.sqrt(1 + x ** 2 + y ** 2)
        cos_theta = np.sqrt(1 + x ** 2) / np.sqrt(1 + x ** 2 + y ** 2)
        R_x = np.array([[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]], dtype=np.float32)
        sin_phi = x / np.sqrt(1 + x ** 2)
        cos_phi = 1 / np.sqrt(1 + x ** 2)
        R_y = np.array([[cos_phi, 0, sin_phi], [0, 1, 0], [-sin_phi, 0, cos_phi]], dtype=np.float32)
        self.R = R_y @ R_x

        # update focal length for virtual camera; please refer to the paper "PCLs: Geometry-aware Neural Reconstruction of 3D Pose with Perspective Crop Layers" for more details.
        if option == 'length':
            self.K[0, 0] = self.K[0, 0] * np.sqrt(1 + x ** 2 + y ** 2)
            self.K[1, 1] = self.K[1, 1] * np.sqrt(1 + x ** 2 + y ** 2)
        
        if option == 'scale':
            self.K[0, 0] = self.K[0, 0] * np.sqrt(1 + x ** 2 + y ** 2) * np.sqrt(1 + x ** 2)
            self.K[1, 1] = self.K[1, 1] * (1 + x ** 2 + y ** 2)/ np.sqrt(1 + x ** 2)

    def update_intrinsics_after_crop(self, bbox):
        left, upper, _, _ = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_intrinsics_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    def update_intrinsics_after_scale(self, scale_factor):
        self.K[0, 0] /= scale_factor
        self.K[1, 1] /= scale_factor

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def intrinsics(self):
        return self.K

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])