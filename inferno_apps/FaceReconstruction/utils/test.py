import random
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
def test(model, batch, delta=None):
    def _to_list(x):
        # Accepts tensor/ndarray/list; returns a clean Python list rounded to 4 decimals
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if x.ndim > 1:
                x = x[0]  # show the first sample
            x = x.tolist()
        elif hasattr(x, "tolist"):
            x = x.tolist()
        return [round(float(v), 4) for v in x]

    batch["image"] = batch["image"].cuda()

    print(f"Type of batch: {type(batch)}")
    for key, value in batch.items():
        print(f"Key: {key}, Type: {type(value)}")
    if len(batch["image"].shape) == 3:
        batch["image"] = batch["image"].view(1,3,224,224)

    values = model(batch, training=False, validation=False)
    if "globalpose" in values:
        print(f"Type of globalpose: {type(values['globalpose'])}, shape: {values['globalpose'].shape}, values['globalpose'] = {values['globalpose']} ")
    print("-----------------------------------------")
    org_gp = values.get('globalpose', None)
    original_mask = values['predicted_mask']

    # normalize delta -> float
    if isinstance(delta, torch.Tensor):
        delta_value = float(delta.item())
    else:
        delta_value = float(delta)

    # three rotation deltas
    delta_poses = [
        torch.tensor([delta_value, 0.0, 0.0], dtype=original_mask.dtype, device=original_mask.device), # x
        torch.tensor([0.0, delta_value, 0.0], dtype=original_mask.dtype, device=original_mask.device), # y
        torch.tensor([0.0, 0.0, delta_value], dtype=original_mask.dtype, device=original_mask.device), # z
    ]
    mask_titles = ["Original", "GlobalPose=0", "X Rotated", "Y Rotated", "Z Rotated", "Assigned Pose "]

    visdict_list = []
    pose_list = []  # <<< will store the pose vector actually used for each visualization

    # --- original (no rotation) ---
    visdict = model.visualize_landmarks_pred_fan(batch, None)
    visdict_list.append(visdict)
    if "globalpose" in values:
        pose_list.append(_to_list(values["globalpose"][0]))
    else:
        pose_list.append(["N/A"])

    # --- globalpose = 0 ---
    if "globalpose" in values:
        enc_like_zero = {k: values[k] for k in ["shapecode", "expcode", "jawpose", "globalpose", "cam", "lightcode", "texcode"] if k in values}
        gp = enc_like_zero["globalpose"]
        zero_pose = torch.zeros_like(gp)
        enc_like_zero["globalpose"] = zero_pose
        flame_zero = model.shape_model(enc_like_zero)
        new_values_zero = model.renderer({**enc_like_zero, **flame_zero})
        # update batch keys present in both
        common_keys = set(batch.keys()) & set(new_values_zero.keys())
        for key in common_keys:
            batch[key] = new_values_zero[key]
        if isinstance(batch["image"], torch.Tensor):
            batch["image"] = batch["image"].detach().cpu()
        visdict_zero = model.visualize_landmarks_pred_fan(batch, None)
        visdict_list.append(visdict_zero)
        pose_list.append(_to_list(zero_pose[0]))
    else:
        visdict_list.append(None)
        pose_list.append(["N/A"])

    # --- for each axis rotation ---
    for d_pose in delta_poses:
        if "globalpose" in values:
            values["cam"] = torch.tensor([[ 0.5e+01, 0,  1.0415e-01]]).to(device=gp.device, dtype=torch.float32)
            enc_like = {k: values[k] for k in ["shapecode", "expcode", "jawpose", "globalpose", "cam", "lightcode", "texcode"] if k in values}
            gp = enc_like["globalpose"]
            delta_pose = d_pose.to(gp.device).view(1, 3).repeat(gp.size(0), 1)
            enc_like["globalpose"] = delta_pose
            print(f"Applying delta_pose: {delta_pose}, now globalpose = {enc_like['globalpose']}")
            flame = model.shape_model(enc_like)
            new_values = model.renderer({**enc_like, **flame})

            # update batch keys present in both
            if 'new_values' in locals():
                common_keys = set(batch.keys()) & set(new_values.keys())
                for key in common_keys:
                    batch[key] = new_values[key]
            if isinstance(batch["image"], torch.Tensor):
                batch["image"] = batch["image"].detach().cpu()

            visdict = model.visualize_landmarks_pred_fan(batch, None)
            visdict_list.append(visdict)
            pose_list.append(_to_list(enc_like["globalpose"][0]))
        else:
            visdict_list.append(None)
            pose_list.append(["N/A"])

    # --- for assigning specific poses ---
    R = np.array([[0.8713, 0, -0.4907],
                        [0.1338, 0.9621, 0.2375],
                        [0.4721, -0.2726, 0.8383]])
    rvec, _ = cv2.Rodrigues(R)
    
    rvec = np.array([[3.14/2,3.14/2,0]])
    rvec = np.array([[3.14/2,3.14/2,3.14/2]])
    rvec = np.array([[3.14/2-0.1,3.14/2-0.1,3.14/2-0.1]])
    # rvec = np.array([[-0.27000075],
    #                 [0.07082269],
    #                 [-0.050961195 ]])
    assigned_poses = torch.from_numpy(rvec).to(device=gp.device, dtype=torch.float32).view(-1, 3)   # example specific pose
    
    for a_pose in assigned_poses:
        if "globalpose" in values:
            #enc_like cam = tensor([[ 1.0465e+01, -5.1096e-04,  3.0415e-02]], device='cuda:0')
            values["cam"] = torch.tensor([[ 0.5e+01, -1.0415e-01,  2.0415e-01]]).to(device=gp.device, dtype=torch.float32)
            enc_like = {k: values[k] for k in ["shapecode", "expcode", "jawpose", "globalpose", "cam", "lightcode", "texcode"] if k in values}
            print(f"enc_like cam = {values['cam']}")
            gp = enc_like["globalpose"]
            assigned_pose = a_pose.to(gp.device).view(1, 3).repeat(gp.size(0), 1)
            enc_like["globalpose"] = assigned_pose
            print(f"Applying assigned_pose: {assigned_pose}, now globalpose = {enc_like['globalpose']}")
            flame = model.shape_model(enc_like)
            new_values = model.renderer({**enc_like, **flame})

            # update batch keys present in both
            if 'new_values' in locals():
                common_keys = set(batch.keys()) & set(new_values.keys())
                for key in common_keys:
                    batch[key] = new_values[key]
            if isinstance(batch["image"], torch.Tensor):
                batch["image"] = batch["image"].detach().cpu()

            visdict = model.visualize_landmarks_pred_fan(batch, None)
            visdict_list.append(visdict)
            pose_list.append(_to_list(enc_like["globalpose"][0]))
        else:
            visdict_list.append(None)
            pose_list.append(["N/A"])
    

    # --- render grid with pose in titles ---
    n = len(visdict_list)
    fig, axs = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axs = axs.reshape(2, 1)

    for i, visdict in enumerate(visdict_list):
        title_core = mask_titles[i]
        pose_str = f"pose={pose_list[i]}" if isinstance(pose_list[i], list) else "pose=N/A"
        if visdict is not None and "landmarks_pred_fan" in visdict and "shape_image" in visdict:
            img0 = visdict["landmarks_pred_fan"][0]
            img1 = visdict["shape_image"][0]
            if img0.ndim == 4 and img0.shape[0] == 1:
                img0 = img0.squeeze(0)
            if img1.ndim == 4 and img1.shape[0] == 1:
                img1 = img1.squeeze(0)

            axs[0][i].imshow(img0)
            axs[0][i].set_title(f"landmarks_pred_fan — {title_core}\n{pose_str}", fontsize=10)
            axs[0][i].axis('off')

            axs[1][i].imshow(img1)
            axs[1][i].set_title(f"shape_image — {title_core}\n{pose_str}", fontsize=10)
            axs[1][i].axis('off')
        else:
            axs[0][i].set_visible(False)
            axs[1][i].set_visible(False)

    plt.suptitle("Comparison for all rotations (with corresponding global pose)", y=1.02)
    plt.tight_layout()
    plt.show()
# TAG:batch_globalpose_restore

    batch["globalpose"] = org_gp  # restore original globalpose
    #atch["globalpose"] = zero_pose # set to zero pose for consistency
    #print shapecode and expcode

    return values

#--------------------------------------------------------------------------------------------------------------



# def test(model, batch):
#     batch["image"] = batch["image"].cuda()
#     if len(batch["image"].shape) == 3:
#         batch["image"] = batch["image"].view(1,3,224,224)
#     values = model(batch, training=False, validation=False)
#     return values




#----------------------------------------------------------------------------------------------------------------------------
# import random
# import torch
# import matplotlib.pyplot as plt

# def test(model, batch, delta_pose=None):
#     batch["image"] = batch["image"].cuda()
    
#     print(f"Type of batch: {type(batch)}")
#     #batch is a dictionary, print its keys and types
#     for key, value in batch.items():    
#         print(f"Key: {key}, Type: {type(value)}")
#     if len(batch["image"].shape) == 3:
#         batch["image"] = batch["image"].view(1,3,224,224)
    
#     # 讓整個模型照原本的 forward 跑（它知道如何處理各維度和鍵）
#     values = model(batch, training=False, validation=False)
#     print(f"Type of globalpose: {type(values['globalpose'])}, shape: {values['globalpose'].shape}, values['globalpose'] = {values['globalpose']} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     print("-----------------------------------------")
#     print(f"Type of batch: {type(batch)}")
#     #batch is a dictionary, print its keys and types
#     for key, value in batch.items():    
#         print(f"Key: {key}, Type: {type(value)}")


#     original_mask = values['predicted_mask']
    

#     # 若 values 內有輸出參數（例如 expcode/shape/globalpose/cam 等）或 mesh，
#     # 你可以在這裡覆寫 globalpose 後，用 model.renderer 重新渲染一次：
#     if delta_pose is not None and "globalpose" in values:
#         enc_like = {k: values[k] for k in ["shapecode", "expcode", "jawpose", "globalpose", "cam", "lightcode", "texcode"] if k in values}
#         gp = enc_like["globalpose"]
#         delta = delta_pose.to(gp.device).view(1, 3).repeat(gp.size(0), 1)
#         enc_like["globalpose"] = gp + delta
#         flame = model.shape_model(enc_like)
#         new_values = model.renderer({**enc_like, **flame})


#     # Compare batch and new_values for common keys, and update batch only if key exists in both
#     if 'new_values' in locals():
#         common_keys = set(batch.keys()) & set(new_values.keys())
#         for key in common_keys:
#             batch[key] = new_values[key]


#     new_mask = new_values['predicted_mask']
#     # #compare original_mask and new_mask with visualizationss
#     def show_masks(mask1, mask2, title1="Original Mask", title2="New Mask"):
#         fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#         axs[0].imshow(mask1.squeeze().detach().cpu().numpy(), cmap='gray')
#         axs[0].set_title(title1)
#         axs[0].axis('off')
#         axs[1].imshow(mask2.squeeze().detach().cpu().numpy(), cmap='gray')
#         axs[1].set_title(title2)
#         axs[1].axis('off')
#         plt.tight_layout()
#         plt.show()

#     show_masks(original_mask, new_mask)

#     return new_values