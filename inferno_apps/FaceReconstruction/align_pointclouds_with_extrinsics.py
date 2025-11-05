#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 front / left / right 三個點雲用外參旋到同一座標系，並嘗試讓臉重疊。
預設只用旋轉 (Xw = R^T Xc) + 以重心對齊至 front 的重心（避免 t 與尺度不一致）。

用法：
  python align_pointclouds_with_extrinsics.py multiview \
      --poses poses.json \
      --rotation-only         # 預設；僅用旋轉，之後以重心對齊 front
  # 或
  python align_pointclouds_with_extrinsics.py multiview \
      --poses poses.json \
      --use-translation --scale 1.0  # 若你確認 t 的單位與點雲相容

輸入資料夾：
  mergeRGBdepth/<name>_resize_result/
應包含：
  front_rgbd_pointcloud.ply
  left_rgbd_pointcloud.ply
  right_rgbd_pointcloud.ply
以及 poses.json（下方提供可直接貼的 JSON）
"""

import json
import argparse
from pathlib import Path
import numpy as np

# ----------------- 簡單 ASCII PLY 讀寫 -----------------
def load_ply_ascii_xyzrgb(ply_path: Path):
    with open(ply_path, "r", encoding="utf-8") as f:
        header = []
        line = f.readline().strip()
        if line != "ply":
            raise ValueError(f"{ply_path} 不是 PLY 檔（開頭非 ply）")
        header.append(line)
        num_vertices = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("不完整的 PLY 檔")
            line = line.strip()
            header.append(line)
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            if line == "end_header":
                break
        if num_vertices is None:
            raise ValueError("PLY header 沒有 element vertex")
        data = []
        for _ in range(num_vertices):
            parts = f.readline().strip().split()
            if len(parts) < 6:
                raise ValueError("PLY 點格式需至少 x y z r g b")
            x, y, z = map(float, parts[:3])
            r, g, b = map(int, parts[3:6])
            data.append((x, y, z, r, g, b))
        arr = np.array(data, dtype=np.float32)
        xyz = arr[:, :3].astype(np.float32)
        rgb = arr[:, 3:].astype(np.uint8)
        return xyz, rgb

def save_ply_ascii_xyzrgb(ply_path: Path, xyz: np.ndarray, rgb: np.ndarray):
    N = xyz.shape[0]
    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ])
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

# ----------------- 外參處理 -----------------
def world_from_camera(R, t, Xc, use_translation=False, scale=1.0):
    """
    R: (3,3) world->cam
    t: (3,)  in camera coords (滿足 Xc = R Xw + t)
    Xc: (N,3) camera-frame points
    use_translation=False: 僅旋轉 -> Xw = R^T Xc
    use_translation=True : Xw = R^T (s*Xc - t)  # s 是你的點雲座標相對於 t 的尺度
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3,)
    Xc = np.asarray(Xc, dtype=np.float64)
    if use_translation:
        return (R.T @ (scale * Xc - t).T).T
    else:
        return (R.T @ Xc.T).T

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", default="multiview", help="資料夾名稱（不含 mergeRGBdepth 與 _resize_result）")
    ap.add_argument("--poses", type=str, default="poses_siedse.json", help="包含 front/left/right 外參的 JSON 檔路徑")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rotation-only", action="store_true", help="只用旋轉（預設），再以重心對齊 front")
    g.add_argument("--use-translation", action="store_true", help="使用 t（需確定尺度一致）")

    ap.add_argument("--invert-rot", default= True, action="store_true", help="把所有旋轉方向反向（用 R^T）")

    ap.add_argument("--scale", type=float, default=1.0, help="當使用 t 時，給相機座標的尺度係數（預設 1.0）")
    ap.add_argument("--views", nargs="+", default=["front", "left", "right"], help="要處理的視角清單")

    # Added: allow user to apply X-axis shifts (world coordinates) to left/right clouds after alignment
    ap.add_argument("--shift-left", type=float, default=0.0, help="額外沿世界 X 軸平移 left 點雲（加入 centroid 對齊後）")
    ap.add_argument("--shift-right", type=float, default=0.0, help="額外沿世界 X 軸平移 right 點雲（加入 centroid 對齊後）")

    # 新增：允許對 left/right 做微小的 Y 軸（yaw）手動調整，單位為度
    ap.add_argument("--yaw-left", type=float, default=0.0, help="以度為單位，沿世界 Y 軸對 left 點雲做微小旋轉（正為右手規則）")
    ap.add_argument("--yaw-right", type=float, default=0.0, help="以度為單位，沿世界 Y 軸對 right 點雲做微小旋轉（正為右手規則）")

    # 新增：移除所有 RGB=(0,0,0) 的黑色點
    ap.add_argument("--remove-black", action="store_true", help="移除所有顏色為 (0,0,0) 的點")

    # 新增：對 left/right 使用相同 magnitude 的對稱參數（若給定，會覆蓋個別的 --shift-left/--shift-right / --yaw-left/--yaw-right / --yaw-right）
    ap.add_argument("--symmetric-shift", type=float, default=0.0, help="對 left/right 使用相同的 X 軸平移量 s，left=+s, right=-s（會覆蓋 --shift-left/--shift-right）")
    ap.add_argument("--symmetric-yaw", type=float, default=0.0, help="對 left/right 使用相同的 yaw（Y 軸）量 y（度），left=-y, right=+y（會覆蓋 --yaw-left/--yaw-right）")
    ap.add_argument("--symmetric-zrot", type=float, default=0.0, help="對 left/right 使用相同的 Z 軸旋轉量 z（度），left=-z, right=+z")

    # 新增：讓 front 在完成旋轉/對齊後沿世界 Z 軸平移（正值為 +Z）
    ap.add_argument("--shift-front-z", type=float, default=0.0, help="在完成對齊後，沿世界 Z 軸平移 front 點雲（正為 +Z）")

    # 新增：讓 front 在完成對齊後沿世界 X 軸旋轉（度）
    ap.add_argument("--rot-front-x", type=float, default=0.0, help="在完成對齊後，繞世界 X 軸旋轉 front 點雲（度，正為右手規則）")

    # 新增：讓 front 在完成對齊後沿世界 Y 軸平移（正值為 +Y）
    ap.add_argument("--shift-front-y", type=float, default=0.0, help="在完成對齊後，沿世界 Y 軸平移 front 點雲（正為 +Y）")
    ap.add_argument("--remove-left-neg-right-pos", action="store_true",
                    help="在完成對齊後，移除 left 中 X < 0 與 right 中 X > 0 的點（會同步更新 RGB）")

    return ap.parse_args()

def main():
    args = parse_args()
    folder = Path(f"mergeRGBdepth/{args.name}_resize_result")
    if not folder.exists():
        raise FileNotFoundError(f"找不到資料夾：{folder}")

    # 讀 poses
    if args.poses is not None:
        poses_path = Path(args.poses)
        with open(poses_path, "r", encoding="utf-8") as f:
            poses_all = json.load(f)
    else:
        # 允許把 JSON 直接貼到此檔最下方並 import；或改成預設路徑
        raise FileNotFoundError("請用 --poses 指定外參 JSON 檔")

    views_info = poses_all["views"]

    # 讀每個視角的點雲
    xyz_cam = {}
    rgb_cam = {}
    for v in args.views:
        ply = folder / f"{v}_rgbd_pointcloud.ply"
        if not ply.exists():
            raise FileNotFoundError(f"缺少點雲：{ply}")
        xyz, rgb = load_ply_ascii_xyzrgb(ply)
        xyz_cam[v] = xyz
        rgb_cam[v] = rgb
        print(f"[LOAD] {v}: {xyz.shape[0]} points")

    # 轉到世界座標（只旋轉或含平移）
    xyz_world = {}
    for v in args.views:
        R_wc = np.array(views_info[v]["pose"]["R"], dtype=np.float64)
        t_c  = np.array(views_info[v]["pose"]["t"], dtype=np.float64)

        if args.invert_rot:
            # 把 world->camera 的 R 全部反向（取 R^T）
            R_wc = R_wc.T

        xyz_world[v] = world_from_camera(R_wc, t_c, xyz_cam[v],
                                        use_translation=args.use_translation,
                                        scale=args.scale).astype(np.float32)

    # 以 front 作為參考，把各雲重心對齊（rotation-only 時特別有用）
    ref = "front" if "front" in args.views else args.views[0]
    ref_centroid = np.nanmean(xyz_world[ref], axis=0)

    for v in args.views:
        c = np.nanmean(xyz_world[v], axis=0)
        xyz_world[v] = xyz_world[v] + (ref_centroid - c)

    # Apply optional X-axis adjustments for left/right views (in world coordinates)
    for v in args.views:
        # symmetric shift takes precedence if provided
        shift_x = 0.0
        if abs(args.symmetric_shift) > 0.0 and v in ("left", "right"):
            shift_x = args.symmetric_shift if v == "left" else -args.symmetric_shift
        else:
            if v == "left":
                shift_x = -args.shift_left
            elif v == "right":
                shift_x = -args.shift_right
        if abs(shift_x) > 0.0:
            xyz_world[v][:, 0] = xyz_world[v][:, 0] + float(shift_x)
            print(f"[SHIFT] Applied X offset {shift_x:.6f} to view '{v}'")

    # 新增：Apply optional yaw (Y-axis) adjustments for left/right views, around reference centroid
    for v in args.views:
        # symmetric yaw takes precedence if provided
        if abs(args.symmetric_yaw) > 0.0 and v in ("left", "right"):
            yaw_deg = -args.symmetric_yaw if v == "left" else args.symmetric_yaw
        else:
            yaw_deg = 0.0
            if v == "left":
                yaw_deg = args.yaw_left
            elif v == "right":
                yaw_deg = args.yaw_right
        if abs(yaw_deg) > 1e-12:
            xyz_world[v] = rotate_points_around_world_y(xyz_world[v], yaw_deg, ref_centroid)
            print(f"[YAW] Rotated view '{v}' around world Y by {yaw_deg:.4f} deg (center={ref_centroid.tolist()})")

    # 新增：Apply optional Z-axis rotations (around world Z), symmetric takes precedence
    for v in args.views:
        if abs(args.symmetric_zrot) > 0.0 and v in ("left", "right"):
            z_deg = -args.symmetric_zrot if v == "left" else args.symmetric_zrot
        else:
            z_deg = 0.0  # no per-view zrot previously existed; keep 0
        if abs(z_deg) > 1e-12:
            xyz_world[v] = rotate_points_around_world_z(xyz_world[v], z_deg, ref_centroid)
            print(f"[ZROT] Rotated view '{v}' around world Z by {z_deg:.4f} deg (center={ref_centroid.tolist()})")

    # 新增：如果指定 --remove-black，過濾掉所有 RGB == (0,0,0) 的點（同步更新 xyz 與 rgb）
    if args.remove_black:
        for v in args.views:
            rgb = rgb_cam[v]
            # mask 保留非全黑的點
            mask = ~((rgb[:, 0] == 0) & (rgb[:, 1] == 0) & (rgb[:, 2] == 0))
            n_before = rgb.shape[0]
            n_after = int(mask.sum())
            removed = n_before - n_after
            if removed > 0:
                xyz_world[v] = xyz_world[v][mask]
                rgb_cam[v] = rgb[mask]
            print(f"[FILTER] view '{v}': removed {removed} black points, {n_after} remain")
            if n_after == 0:
                print(f"[WARN] view '{v}' has no points after removing black points")

    # 新增：在所有旋轉/對齊處理完成後，對 front 做繞世界 X 軸的旋轉（若有指定）
    if abs(args.rot_front_x) > 1e-12 and ref in args.views:
        xyz_world[ref] = rotate_points_around_world_x(xyz_world[ref], args.rot_front_x, ref_centroid)
        print(f"[ROT-X] Rotated view '{ref}' around world X by {float(args.rot_front_x):.4f} deg (center={ref_centroid.tolist()})")

    # 新增：在所有旋轉/對齊處理完成後，對 front 做沿世界 Y 軸的微調（若有指定）
    if abs(args.shift_front_y) > 0.0 and ref in args.views:
        xyz_world[ref][:, 1] = xyz_world[ref][:, 1] + float(args.shift_front_y)
        print(f"[SHIFT] Applied Y offset {args.shift_front_y:.6f} to view '{ref}'")

    # 新增：在所有旋轉/對齊處理完成後，對 front 做沿世界 Z 軸的微調（若有指定）
    if abs(args.shift_front_z) > 0.0 and ref in args.views:
        xyz_world[ref][:, 2] = xyz_world[ref][:, 2] + float(args.shift_front_z)
        print(f"[SHIFT] Applied Z offset {args.shift_front_z:.6f} to view '{ref}'")

    # 新增：如果指定 --remove-left-neg-right-pos，過濾掉 left 中 X < 0 與 right 中 X > 0（在完成對齊後執行，會同步更新 RGB）
    if args.remove_left_neg_right_pos:
        for v in args.views:
            if v == "left":
                # 保留 X <= 0 (移除 X > 0)
                mask = xyz_world[v][:, 0] <= 0.0
                desc = "X > 0"
            elif v == "right":
                # 保留 X >= 0 (移除 X < 0)
                mask = xyz_world[v][:, 0] >= 0.0
                desc = "X < 0"
            else:
                continue
            n_before = xyz_world[v].shape[0]
            n_after = int(mask.sum())
            removed = n_before - n_after
            if removed > 0:
                xyz_world[v] = xyz_world[v][mask]
                rgb_cam[v] = rgb_cam[v][mask]
            print(f"[REMOVE-SIDE] view '{v}': removed {removed} points where {desc}, {n_after} remain")
            if n_after == 0:
                print(f"[WARN] view '{v}' has no points after remove-side-x")

    for v in args.views:
        out_ply = folder / f"{v}_aligned_world.ply"
        save_ply_ascii_xyzrgb(out_ply, xyz_world[v], rgb_cam[v])
        print(f"[SAVE] {out_ply.name}  ({xyz_world[v].shape[0]} pts)")

    # 合併輸出
    merged_xyz = np.concatenate([xyz_world[v] for v in args.views], axis=0)
    merged_rgb = np.concatenate([rgb_cam[v] for v in args.views], axis=0)
    merged_out = folder / "aligned_merged.ply"
    save_ply_ascii_xyzrgb(merged_out, merged_xyz, merged_rgb)
    print(f"[SAVE] {merged_out.name}  ({merged_xyz.shape[0]} pts total)")
    print("[DONE] 點雲已旋轉到同一座標，並以重心對齊。若要用平移 t，請加 --use-translation 並調整 --scale。")

# 新增：繞世界 Y 軸，圍繞中心點旋轉（angle_deg 為度）
def rotate_points_around_world_y(X: np.ndarray, angle_deg: float, center: np.ndarray):
    angle_rad = float(angle_deg) * np.pi / 180.0
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array([[c, 0.0, s],
                  [0.0, 1.0, 0.0],
                  [-s, 0.0, c]], dtype=np.float64)
    Xc = X.astype(np.float64) - center.reshape(1, 3)
    Xr = (R @ Xc.T).T + center.reshape(1, 3)
    return Xr.astype(np.float32)

# 新增：繞世界 Z 軸，圍繞中心點旋轉（angle_deg 為度）
def rotate_points_around_world_z(X: np.ndarray, angle_deg: float, center: np.ndarray):
    angle_rad = float(angle_deg) * np.pi / 180.0
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    Xc = X.astype(np.float64) - center.reshape(1, 3)
    Xr = (R @ Xc.T).T + center.reshape(1, 3)
    return Xr.astype(np.float32)

# 新增：繞世界 X 軸，圍繞中心點旋轉（angle_deg 為度）
def rotate_points_around_world_x(X: np.ndarray, angle_deg: float, center: np.ndarray):
    angle_rad = float(angle_deg) * np.pi / 180.0
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0,  c, -s],
                  [0.0,  s,  c]], dtype=np.float64)
    Xc = X.astype(np.float64) - center.reshape(1, 3)
    Xr = (R @ Xc.T).T + center.reshape(1, 3)
    return Xr.astype(np.float32)

if __name__ == "__main__":
    main()
