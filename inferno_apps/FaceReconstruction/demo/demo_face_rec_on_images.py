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

from inferno_apps.FaceReconstruction.utils.load import load_model
from inferno.datasets.ImageTestDataset import TestData
import inferno
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse
from inferno_apps.FaceReconstruction.utils.output import save_obj, save_images, save_codes
from inferno_apps.FaceReconstruction.utils.test import test
from inferno.utils.other import get_path_to_assets
#from inferno.models.Renderer import render_depth



def main():
    parser = argparse.ArgumentParser()
    # add the input folder arg 
    #parser.add_argument('--input_folder', type=str, default= str(Path(get_path_to_assets())/ "data/EMOCA_test_example_data/images/affectnet_test_examples"))
    parser.add_argument('--input_folder', type=str, default= str(Path(get_path_to_assets())/ "/home/inferno/src/inferno/inferno_apps/FaceReconstruction/demo/TestSamples/ruei4PP"))
    parser.add_argument('--output_folder', type=str, default="demo/TestSamples/ruei4PP", help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMICA-CVT_flame2020_notexture', help='Name of the model to use.')
    # parser.add_argument('--model_name', type=str, default='EMICA_flame2020_notexture', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str, default=str(Path(get_path_to_assets()) / "FaceReconstruction/models"))
    parser.add_argument('--save_images', type=bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=True, help="If true, output meshes will be saved")
    parser.add_argument('--save_all_viskeys', type=bool, default=True, help="If true, save all visdict images in a separate folder.")
    parser.add_argument('--save_depth', type=bool, default=True, help="Save a depth map with the SAME view as the visualization (DECA renderer).")
    args = parser.parse_args()


    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    path_to_models = args.path_to_models
    input_folder = args.input_folder
    output_folder = args.output_folder
    model_name = args.model_name

    # 1) Load the model
    face_rec_model, conf = load_model(path_to_models, model_name)
    face_rec_model.cuda()
    face_rec_model.eval()

    # 2) Create a dataset
    dataset = TestData(input_folder, face_detector="fan", max_detection=20)
    # dataset = TestData(
    #     input_folder,
    #     iscrop=False,          # 關閉裁切
    #     crop_size=224,         # 仍會把整張圖縮成 224x224 輸出
    #     face_detector="fan",
    #     max_detection=20
    # )

    ## 4) Run the model on the data
    for i in auto.tqdm( range(len(dataset))):
        batch = dataset[i]
        vals = test(face_rec_model, batch)
        visdict = face_rec_model.visualize_batch(batch, i, None, in_batch_idx=None)
        print(f"visdict keys: {list(visdict.keys())}")
        # name = f"{i:02d}"
        current_bs = batch["image"].shape[0]

        for j in range(current_bs):
            name =  batch["image_name"][j]
            print("image_name:", name)

            sample_output_folder = Path(output_folder) / name
            sample_output_folder.mkdir(parents=True, exist_ok=True)

            if args.save_mesh:
                save_obj(face_rec_model, str(sample_output_folder / "mesh_coarse.obj"), vals, j)
            if args.save_codes:
                save_codes(Path(output_folder), name, vals, i=j)
            if args.save_images:
                save_images(output_folder, name, visdict, with_detection=True, i=j)
            if args.save_depth:
                try:
                    # Don't overwrite visdict!
                    depth_image = face_rec_model.save_depths(batch, i, None, in_batch_idx=None)
                    print("depth_image dtype:", depth_image.dtype)
                    print("depth_image min:", depth_image.min().item())
                    print("depth_image max:", depth_image.max().item())
                    # Save all depth maps in a subfolder under output_folder
                    depth_maps_folder = os.path.join(output_folder, 'depth_maps')
                    os.makedirs(depth_maps_folder, exist_ok=True)
                    # Save depth map as numpy .npy file (recommended)
                    np.save(os.path.join(depth_maps_folder, name + '_depth.npy'), depth_image[0].detach().cpu().numpy())
                except Exception as e:
                    print(f"[WARN] save depth failed for {name}: {e}")
            
 
            if args.save_all_viskeys:
                all_viskeys_folder = Path(output_folder) / "all vis keys" / name
                all_viskeys_folder.mkdir(parents=True, exist_ok=True)
                for viskey, img in visdict.items():
                    # Handle lists of images (e.g., per batch)
                    if isinstance(img, list):
                        for idx, img_item in enumerate(img):
                            img_np = img_item
                            if isinstance(img_np, torch.Tensor):
                                img_np = img_np.detach().cpu().numpy()
                                if img_np.ndim == 3 and img_np.shape[0] in [1,3]:
                                    img_np = np.transpose(img_np, (1,2,0))
                                if img_np.dtype != np.uint8:
                                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                            # Squeeze channel if shape is (H,W,1)
                            if img_np.ndim == 3 and img_np.shape[2] == 1:
                                img_np = np.squeeze(img_np, axis=2)
                            # If still single channel, convert to 3-channel grayscale
                            if img_np.ndim == 2:
                                img_np = np.stack([img_np]*3, axis=-1)
                            imsave(str(all_viskeys_folder / f"{viskey}_{idx}.png"), img_np)
                    else:
                        img_np = img
                        if isinstance(img_np, torch.Tensor):
                            img_np = img_np[j].detach().cpu().numpy() if img_np.shape[0] == current_bs else img_np.detach().cpu().numpy()
                            if img_np.ndim == 3 and img_np.shape[0] in [1,3]:
                                img_np = np.transpose(img_np, (1,2,0))
                            if img_np.dtype != np.uint8:
                                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                        if img_np.ndim == 3 and img_np.shape[2] == 1:
                            img_np = np.squeeze(img_np, axis=2)
                        if img_np.ndim == 2:
                            img_np = np.stack([img_np]*3, axis=-1)
                        imsave(str(all_viskeys_folder / f"{viskey}.png"), img_np)

    print("Done")


if __name__ == '__main__':
    main()
