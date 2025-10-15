import random
import torch
import matplotlib.pyplot as plt

def test(model, batch, delta=0.2):
    batch["image"] = batch["image"].cuda()
    
    print(f"Type of batch: {type(batch)}")
    #batch is a dictionary, print its keys and types
    for key, value in batch.items():    
        print(f"Key: {key}, Type: {type(value)}")
    if len(batch["image"].shape) == 3:
        batch["image"] = batch["image"].view(1,3,224,224)
    values = model(batch, training=False, validation=False)
    print(f"Type of globalpose: {type(values['globalpose'])}, shape: {values['globalpose'].shape}, values['globalpose'] = {values['globalpose']} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("-----------------------------------------")
    print(f"Type of batch: {type(batch)}")
    #batch is a dictionary, print its keys and types
    # for key, value in batch.items():    
    #     print(f"Key: {key}, Type: {type(value)}")
    original_mask = values['predicted_mask']

    # 確保 delta 是 float
    if isinstance(delta, torch.Tensor):
        delta_value = float(delta.item())
    else:
        delta_value = float(delta)

    # 定義三個旋轉方向
    delta_poses = [
        torch.tensor([delta_value, 0.0, 0.0], dtype=original_mask.dtype, device=original_mask.device), # x
        torch.tensor([0.0, delta_value, 0.0], dtype=original_mask.dtype, device=original_mask.device), # y
        torch.tensor([0.0, 0.0, delta_value], dtype=original_mask.dtype, device=original_mask.device), # z
    ]
    mask_titles = ["Original Mask", "GlobalPose=0", "X Rotated", "Y Rotated", "Z Rotated"]
    masks = [original_mask]

    visdict_list = []

    # 原始（未旋轉）
    visdict = model.visualize_landmarks_pred_fan(batch, None)
    visdict_list.append(visdict)

    # globalpose = 0 比較
    if "globalpose" in values:
        enc_like_zero = {k: values[k] for k in ["shapecode", "expcode", "jawpose", "globalpose", "cam", "lightcode", "texcode"] if k in values}
        gp = enc_like_zero["globalpose"]
        zero_pose = torch.zeros_like(gp)
        enc_like_zero["globalpose"] = zero_pose
        flame_zero = model.shape_model(enc_like_zero)
        new_values_zero = model.renderer({**enc_like_zero, **flame_zero})
        # 更新 batch 內容
        common_keys = set(batch.keys()) & set(new_values_zero.keys())
        for key in common_keys:
            batch[key] = new_values_zero[key]
        if isinstance(batch["image"], torch.Tensor):
            batch["image"] = batch["image"].detach().cpu()
        visdict_zero = model.visualize_landmarks_pred_fan(batch, None)
        visdict_list.append(visdict_zero)
    else:
        visdict_list.append(None)

    # 針對每個旋轉方向計算新 mask
    for idx, d_pose in enumerate(delta_poses):
        if "globalpose" in values:
            enc_like = {k: values[k] for k in ["shapecode", "expcode", "jawpose", "globalpose", "cam", "lightcode", "texcode"] if k in values}
            gp = enc_like["globalpose"]
            delta_pose = d_pose.to(gp.device).view(1, 3).repeat(gp.size(0), 1)
            enc_like["globalpose"] = gp + delta_pose
            flame = model.shape_model(enc_like)
            new_values = model.renderer({**enc_like, **flame})
            if 'new_values' in locals():
                common_keys = set(batch.keys()) & set(new_values.keys())
                for key in common_keys:
                    batch[key] = new_values[key]
            if isinstance(batch["image"], torch.Tensor):
                batch["image"] = batch["image"].detach().cpu()

            visdict = model.visualize_landmarks_pred_fan(batch, None)
            visdict_list.append(visdict)
        else:
            masks.append(torch.zeros_like(original_mask)) # fallback

    # 顯示所有旋轉與原始結果
    n = len(visdict_list)
    fig, axs = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axs = axs.reshape(2, 1)  # 保證可迭代且有正確形狀

    for i, visdict in enumerate(visdict_list):
        if visdict is not None and "landmarks_pred_fan" in visdict and "shape_image" in visdict:
            img0 = visdict["landmarks_pred_fan"][0]
            img1 = visdict["shape_image"][0]
            if img0.ndim == 4 and img0.shape[0] == 1:
                img0 = img0.squeeze(0)
            if img1.ndim == 4 and img1.shape[0] == 1:
                img1 = img1.squeeze(0)
            axs[0][i].imshow(img0)
            axs[0][i].set_title(f"landmarks_pred_fan ({mask_titles[i]})")
            axs[0][i].axis('off')
            axs[1][i].imshow(img1)
            axs[1][i].set_title(f"shape_image ({mask_titles[i]})")
            axs[1][i].axis('off')
    plt.suptitle("Comparison for all rotations")
    plt.tight_layout()
    plt.show()

    return values

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

# # def test(model, batch):
# #     batch["image"] = batch["image"].cuda()
# #     if len(batch["image"].shape) == 3:
# #         batch["image"] = batch["image"].view(1,3,224,224)
# #     values = model(batch, training=False, validation=False)
# #     return values

