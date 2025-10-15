"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""


import glob
from glob import glob
import cv2
import numpy as np
import scipy
import torch
from skimage.io import imread
from skimage.transform import rescale, estimate_transform, warp
from torch.utils.data import Dataset

# from inferno.datasets.FaceVideoDataModule import add_pretrained_deca_to_path
from inferno.datasets.ImageDatasetHelpers import bbox2point
from inferno.utils.FaceDetector import FAN, MediaPipeMeshFD

from pytorch_lightning import LightningDataModule

import os
import mediapipe as mp

class TestDM(LightningDataModule):

    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan',
                 scaling_factor=1.0, max_detection=None):
        super().__init__()
        self.testpath = testpath 
        self.crop = iscrop
        self.crop_size = crop_size
        self.scale = scale
        self.scaling_factor = scaling_factor
        self.face_detector = face_detector

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None):
        self.dataset = TestData(self.testpath, iscrop=self.crop, crop_size=self.crop_size, 
            scale=self.scale, face_detector=self.face_detector,
            scaling_factor=self.scaling_factor, max_detection=None)

    def test_dataloader(self):
        # create a data loader for self.dataset 
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)
        return dataloader


class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan',
                 scaling_factor=1.0, max_detection=None):
        self.max_detection = max_detection
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath):
            self.imagepath_list = glob(testpath + '/*.jpg') + glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
            self.imagepath_list = video2sequence(testpath)
        else:
            print(f'please check the test path: {testpath}')
            exit()
        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.scaling_factor = scaling_factor
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        # add_pretrained_deca_to_path()
        # from decalib.datasets import detectors
        if face_detector == 'fan':
            self.face_detector = FAN()
        elif face_detector == 'mediapipe':
            self.face_detector = MediaPipeMeshFD(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,  # 設 True → 478 點
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
        )
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        #print("Loading image!!!!!!!!!!!!!!!!!!")
        imagepath = str(self.imagepath_list[index])
        imagename = imagepath.split('/')[-1].split('.')[0]

        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        if self.scaling_factor != 1.:
            image = rescale(image, (self.scaling_factor, self.scaling_factor, 1))*255.

        h, w, _ = image.shape

        # 這兩個變數之後要用來畫在裁切圖上  ### NEW/CHANGED
        landmarks = None
        

        if self.iscrop:
            print(f"I am croppinggggggggggggggggggggggggggggggggggggggggggggggggggggggggg")
            # provide kpt as txt file, or mat file (for AFLW2000)
            kpt_matpath = imagepath.replace('.jpg', '.mat').replace('.png', '.mat')
            kpt_txtpath = imagepath.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(kpt_matpath):
                kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
            elif os.path.exists(kpt_txtpath):
                kpt = np.loadtxt(kpt_txtpath)
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
            else:
                # 嘗試取得 landmarks（若偵測器支援）  ### NEW/CHANGED
                try:
                    # 可能回傳 2 或 3 個項目
                    run_out = self.face_detector.run(image)
                    if len(run_out) == 3:
                        print(f"3 items returned from face detector")
                        bbox, bbox_type, landmarks = run_out
                    else:
                        print(f"2 items returned from face detector")
                        bbox, bbox_type = run_out
                        landmarks = None
                except Exception:
                    # 最壞情況：照舊只拿 bbox
                    bbox, bbox_type = self.face_detector.run(image)
                    landmarks = None
                
                if len(bbox) < 1:
                    print('no face detected! run original image')
                    left = 0
                    right = h - 1
                    top = 0
                    bottom = w - 1
                    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
                else:
                    if self.max_detection is None:
                        bbox = bbox[0]
                        left = bbox[0]
                        right = bbox[2]
                        top = bbox[1]
                        bottom = bbox[3]
                        old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
                    else: 
                        old_size, center = [], []
                        num_det = min(self.max_detection, len(bbox))
                        for bbi in range(num_det):
                            bb = bbox[0]
                            left = bb[0]
                            right = bb[2]
                            top = bb[1]
                            bottom = bb[3]
                            osz, c = bbox2point(left, right, top, bottom, type=bbox_type)
                        old_size += [osz]
                        center += [c]
            used_bbox_for_vis = [left, top, right, bottom]

            
            if isinstance(old_size, list):
                size = []
                src_pts = []
                for i in range(len(old_size)):
                    size += [int(old_size[i] * self.scale)]
                    src_pts += [np.array(
                        [[center[i][0] - size[i] / 2, center[i][1] - size[i] / 2], [center[i][0] - size[i] / 2, center[i][1] + size[i] / 2],
                        [center[i][0] + size[i] / 2, center[i][1] - size[i] / 2]])]
            else:
                size = int(old_size * self.scale)
                src_pts = np.array(
                    [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2]])
            # Save the old size and center in the text file
            print(f"old_size: {size}, center: {center}")
            # Save all sizes and centers to a single file in the image folder
            all_size_center_path = os.path.join(os.path.dirname(imagepath), 'all_size_center.txt')
            entry = f"{imagename}: old_size: {np.squeeze(size)}, center: [{np.squeeze(center)[0]},{np.squeeze(center)[1]}]\n"
            # Read existing entries if the file exists
            existing_entries = {}
            if os.path.exists(all_size_center_path):
                with open(all_size_center_path, 'r') as f:
                    for line in f:
                        name, data = line.split(":", 1)
                        existing_entries[name.strip()] = data.strip()
            # Update or add the current entry
            existing_entries[imagename] = f"old_size: {np.squeeze(size)}, center: [{np.squeeze(center)[0]},{np.squeeze(center)[1]}]"
            # Write all entries back to the file
            with open(all_size_center_path, 'w') as f:
                for name, data in existing_entries.items():
                    f.write(f"{name}: {data}\n")


        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
        
        image = image / 255.
        if not isinstance(src_pts, list):
            
            DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)



            # warp 後的裁切圖（HxWxC, float01），同時保留一份給可視化  ### NEW/CHANGED
            dst_image_float = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp), order=3)
            vis_rgb = _ensure_uint8_rgb(dst_image_float)

            # 若有 landmarks，把座標投影到裁切座標系再畫點存圖  ### NEW/CHANGED
            
            # show the bbox and landmarks on the cropped image
            save_dir = os.path.join(os.path.dirname(imagepath), 'crops_with_landmarks')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{imagename}_crop_lm_mediapipe.jpg')

            drew = None
            if landmarks is not None:
                lm2d = _normalize_landmarks_to_xy(landmarks)
            else:
                lm2d = None

            if lm2d is not None and len(lm2d) > 0:
                print(f"Landmarks (normalized) for {imagename}: shape={lm2d.shape}")
                lm2d_crop = tform(lm2d)  # (N,2) -> 投影到裁切後
                drew = _draw_points(vis_rgb, lm2d_crop, color_bgr=(0, 255, 0), r=2, thickness=-1)
            if drew is None:
                print(f"No landmarks or bbox for {imagename}, saving plain crop.")
                # 萬一都沒有，就至少存一張裁切圖
                drew = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

            cv2.imwrite(save_path, drew)
            print(f"Overlay image saved at: {save_path}")  # Print the path and confirmation

            dst_image = dst_image_float.transpose(2, 0, 1)
            return {
                'image': torch.tensor(dst_image).float(),
                'image_name': imagename,
                'image_path': imagepath,
                'overlay_path': save_path,   # 讓你在外面知道檔案存哪  ### NEW/CHANGED
            }
        else:
            
            DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
            dst_images = []
            for i in range(len(src_pts)):
                tform = estimate_transform('similarity', src_pts[i], DST_PTS)
                dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp), order=3)
                dst_image = dst_image.transpose(2, 0, 1)
                dst_images += [dst_image]
            dst_images = np.stack(dst_images, axis=0)

            imagenames = [imagename] * dst_images.shape[0]
            imagepaths = [imagepath] * dst_images.shape[0]
            return {'image': torch.tensor(dst_images).float(),
                    'image_name': imagenames,
                    'image_path': imagepaths,
                    # 'tform': tform,
                    # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                    }




def video2sequence(video_path):
    videofolder = video_path.split('.')[0]
    util.check_mkdir(videofolder)
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = '{}/{}_frame{:04d}.jpg'.format(videofolder, video_name, count)
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list
# ====== 新增：小工具函式 ======  ### NEW/CHANGED
def _ensure_uint8_rgb(img_float01):
    """img_float01: HxWxC, float [0,1]; return uint8 RGB"""
    img = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def _draw_points(img_rgb_uint8, pts_xy, color_bgr=(0, 255, 0), r=2, thickness=-1):
    """在 RGB 圖上畫點；回傳 BGR（for cv2.imwrite）"""
    vis = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    if pts_xy is not None and len(pts_xy) > 0:
        pts_xy = np.asarray(pts_xy).astype(np.float32)
        for x, y in pts_xy:
            cv2.circle(vis, (int(round(x)), int(round(y))), r, color_bgr, thickness)
    return vis

def _draw_bbox(img_rgb_uint8, bbox, color_bgr=(255, 0, 0), thickness=2):
    vis = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, thickness)
    # 也標些四角點（若沒有 landmarks）
    for (x, y) in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]:
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
    return vis
# =================================
def _normalize_landmarks_to_xy(landmarks):
    """
    將各種可能的 landmarks 格式轉成 (N,2) 的像素座標。
    支援：list[(68,2 or 3)], ndarray(68,2/3), ndarray(Nfaces,68,2/3), 以及多餘 singleton 維度。
    """
    if landmarks is None:
        return None
    arr = np.asarray(landmarks)

    # 常見情況：
    # - list of (68,2): 取第一個
    if isinstance(landmarks, (list, tuple)):
        if len(landmarks) == 0:
            return None
        arr = np.asarray(landmarks[0])  # 先拿第一張臉

    # 如果是 (Nfaces, 68, 2/3) -> 取第一張臉
    if arr.ndim == 3:
        arr = arr[0]

    # 若還有多的維度（例如 (68,2,1) 或 (1,68,2)），壓掉 singleton 維度
    arr = np.squeeze(arr)

    # 只留 xy
    if arr.shape[-1] >= 2:
        arr = arr[..., :2]
    else:
        return None

    # 現在應該是 (68,2)，用 float32
    arr = arr.astype(np.float32)

    # 安全檢查，如果還不是 (N,2) 就攤平成 (-1,2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        arr = arr.reshape(-1, 2)

    return arr