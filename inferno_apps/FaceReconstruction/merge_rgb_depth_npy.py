#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
對資料夾內每一張 *_original.(png|jpg|jpeg) 與對應的 *_fullresolution.(npy|npz)
逐一輸出彩色 PLY：<view>_rgbd_pointcloud.ply

使用方式：
  python merge_rgb_depth_npy.py multiview
會掃描：mergeRGBdepth/multiview_resize_result
"""
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# ========== CLI 與資料夾 ==========
ap = argparse.ArgumentParser(description="Merge RGB and depth npy/npz into per-view PLYs")
ap.add_argument("input_folder", help="資料夾名稱（不含 mergeRGBdepth 與 _resize_result）")
ap.add_argument("--front-z-shift", type=float, default=0.0,
                help="對 front view 的點雲在輸出前沿 Z 軸加上此偏移（同單位，於縮放後套用）")
args = ap.parse_args()

folder = Path(f"mergeRGBdepth/{args.input_folder}_resize_result")
if not folder.exists():
    raise FileNotFoundError(f"找不到資料夾：{folder}")

print(f"[INFO] 目標資料夾：{folder.resolve()}")

# ========== 公用函式 ==========
def imread_rgb_keep3(path: Path) -> np.ndarray:
    """讀圖並確保回傳 (H,W,3) uint8；若是 4 通道會去掉 alpha。"""
    img = Image.open(path).convert("RGBA")  # 先 RGBA，方便處理 alpha
    arr = np.asarray(img)
    if arr.ndim == 2:  # 灰階
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]  # 丟 alpha
    return arr.astype(np.uint8)

def load_depth_from_numpy(path: Path, H: int, W: int) -> np.ndarray:
    """
    支援 .npy / .npz，嘗試把深度對齊到 (H,W)：
      - 若本來就是 (H,W) 直接回傳
      - 若 (W,H) 嘗試轉置/旋轉
      - 若 1D 且長度 H*W -> reshape
      - 若其他 2D 尺寸：先嘗試旋轉匹配；再不行就雙線性 resize
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        npz = np.load(path, allow_pickle=False)
        key_candidates = ["depth", "data", "arr_0"]
        key = next((k for k in key_candidates if k in npz), None)
        if key is None:
            if len(npz.files) == 0:
                raise ValueError(f"{path} 為空的 .npz")
            key = npz.files[0]
        arr = npz[key]
    else:
        raise ValueError(f"不支援的副檔名：{path.suffix}")

    arr = np.asarray(arr)

    # (H,W,1) -> (H,W)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    # 合法：已是 (H,W)
    if arr.ndim == 2 and arr.shape == (H, W):
        return arr.astype(np.float32)

    # (W,H) -> 轉置/旋轉試試
    if arr.ndim == 2 and arr.shape == (W, H):
        return np.rot90(arr).astype(np.float32)

    # 1D 向量 -> reshape
    if arr.ndim == 1 and arr.size == H * W:
        return arr.reshape(H, W).astype(np.float32)

    # 其他 2D 尺寸：先旋轉比對
    if arr.ndim == 2:
        arr_rot = np.rot90(arr)
        if arr_rot.shape == (H, W):
            return arr_rot.astype(np.float32)
        # 最後手段：resize 到 (W,H)
        # PIL 需要單通道 'F' 模式處理 float32
        arr_f = arr.astype(np.float32)
        im = Image.fromarray(arr_f)  # 會是 'F' 或 'I;16'
        im_resized = im.resize((W, H), resample=Image.BILINEAR)
        arr_rs = np.array(im_resized, dtype=np.float32)
        return arr_rs

    raise ValueError(
        f"無法把深度形狀 {arr.shape} 對齊到 (H,W)=({H},{W})；請檢查內容。"
    )

def auto_depth_to_meters_like(depth: np.ndarray) -> np.ndarray:
    """
    嘗試把各種量級的深度，轉成 0.3~3m 的近似公尺範圍（僅為視覺/相對尺度用）。
    """
    d = depth.astype(np.float32).copy()
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        raise ValueError("深度沒有有效值 (>0)")
    dmin = float(np.percentile(d[valid], 2))
    dmax = float(np.percentile(d[valid], 98))

    if dmax <= 1.5:
        z_min, z_max = 0.3, 3.0
        Z = (d - dmin) / max(dmax - dmin, 1e-6) * (z_min + (z_max - z_min)) + z_min - (z_min)
        Z = (d - dmin) / max(dmax - dmin, 1e-6) * (3.0 - 0.3) + 0.3
    elif dmax <= 255:
        z_min, z_max = 0.3, 3.0
        d01 = np.clip((d - dmin) / max(dmax - dmin, 1e-6), 0, 1)
        Z = d01 * (z_max - z_min) + z_min
    elif dmax <= 65535:
        Z_mm = d
        def z_from_mm(mm, inv=False):
            mmv = np.maximum(mm, 1e-6)
            z = (1.0 / mmv) if inv else mmv
            return z * (0.001 if not inv else 1.0)
        Z1 = z_from_mm(Z_mm, inv=False)
        Z2 = z_from_mm(Z_mm, inv=True)
        r1 = float(np.percentile(Z1[valid], 98) - np.percentile(Z1[valid], 2))
        r2 = float(np.percentile(Z2[valid], 98) - np.percentile(Z2[valid], 2))
        Z = Z1 if r1 <= r2 else Z2
        zmin = float(np.percentile(Z[valid], 2))
        zmax = float(np.percentile(Z[valid], 98))
        Z = (Z - zmin) / max(zmax - zmin, 1e-6)
        Z = Z * (3.0 - 0.3) + 0.3
    else:
        zmin = float(np.percentile(d[valid], 2))
        zmax = float(np.percentile(d[valid], 98))
        Z = (d - zmin) / max(zmax - zmin, 1e-6)
        Z = Z * (3.0 - 0.3) + 0.3

    Z[~valid] = np.nan
    return Z

def save_ply_with_color(ply_path: Path, points_xyz: np.ndarray, colors_rgb: np.ndarray):
    N = points_xyz.shape[0]
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
        for (x, y, z), (r, g, b) in zip(points_xyz, colors_rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

# ========== 找出所有 view（以 *_original.* 為準） ==========
rgb_files = []
for ext in ("png", "jpg", "jpeg"):
    rgb_files += list(folder.glob(f"*__original.{ext}"))  # 舊少數人會用雙底線
    rgb_files += list(folder.glob(f"*_original.{ext}"))

# 若資料夾沒有 *_original.*，退而求其次用任何 .png/.jpg/.jpeg 當 RGB
fallback_imgs = []
if not rgb_files:
    for ext in ("png", "jpg", "jpeg"):
        fallback_imgs += list(folder.glob(f"*.{ext}"))
    # 排除明顯是結果圖的（如 *_fullresolution_img.jpg、*_overlap.jpg）
    rgb_files = [p for p in fallback_imgs if "_fullresolution" not in p.stem and "_overlap" not in p.stem]

if not rgb_files:
    raise FileNotFoundError("資料夾內找不到任何 RGB 影像（*_original.(png|jpg|jpeg) 或一般圖片）")

# ========== 每一張圖分別處理 ==========
Z_mul = 0.15           # 深度縮放係數（你原本的參數）
target_size = 1.0      # 最終點雲最大邊等比縮放到 1.0（方便可視）

def find_depth_for_view(view_name: str) -> Path:
    """
    依 view 名稱找對應深度檔：
      優先：<view>_fullresolution.(npy|npz) / <view>_rot90_fullresolution.(npy|npz)
      次選：任一包含 view 名的 .npy/.npz
      最後：資料夾內任一 .npy/.npz
    """
    candidates = []
    patterns = [
        f"{view_name}_fullresolution.npy",
        f"{view_name}_fullresolution.npz",
        f"{view_name}_rot90_fullresolution.npy",
        f"{view_name}_rot90_fullresolution.npz",
    ]
    for pat in patterns:
        p = folder / pat
        if p.exists():
            return p

    # 包含 view 名稱的
    name_hits = list(folder.glob(f"*{view_name}*.npy")) + list(folder.glob(f"*{view_name}*.npz"))
    if name_hits:
        return name_hits[0]

    # 任何深度
    any_depth = list(folder.glob("*.npy")) + list(folder.glob("*.npz"))
    if any_depth:
        return any_depth[0]

    raise FileNotFoundError(f"找不到對應深度檔（含 {view_name} 的 .npy/.npz）")

processed = 0
for rgb_path in sorted(set(rgb_files)):
    # 取 view 名（去掉尾巴的 _original）
    stem = rgb_path.stem
    if stem.endswith("_original"):
        view = stem[:-9]  # 去掉 "_original"
    elif stem.endswith("__original"):
        view = stem[:-10]
    else:
        view = stem

    try:
        depth_path = find_depth_for_view(view)
    except FileNotFoundError as e:
        print(f"[WARN] {e} -> 跳過 {rgb_path.name}")
        continue

    print(f"\n[VIEW] {view}")
    print(f"  RGB : {rgb_path.name}")
    print(f"  DEP : {depth_path.name}")

    # ---- 載入 RGB
    rgb = imread_rgb_keep3(rgb_path)  # (H,W,3) uint8
    H, W = rgb.shape[:2]
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    # ---- 載入/對齊深度 -> (H,W) float32
    depth_raw = load_depth_from_numpy(depth_path, H, W).astype(np.float32)
    print(f"  深度形狀: {tuple(depth_raw.shape)} | RGB形狀: {(H, W)}")
    print(f"  原始深度 min={np.nanmin(depth_raw):.6f}, max={np.nanmax(depth_raw):.6f}")

    # ---- 正規化到近似公尺
    Z = auto_depth_to_meters_like(depth_raw)
    print(f"  正規化後 Z min={np.nanmin(Z):.6f}, max={np.nanmax(Z):.6f}")

    # ---- 簡化回投影
    f = float(max(W, H))
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)  # (H,W)
    valid = np.isfinite(Z) & (Z > 0)

    # 座標（X,Y）單位以像素歸一（除以 f），Z 乘縮放並取負（與你原先一致）
    X = (xx - cx) / f
    Y = (yy - cy) / f
    Z = Z * Z_mul * -1.0

    if np.any(valid):
        print(f"  X 範圍: {np.nanmin(X[valid]):.6f} ~ {np.nanmax(X[valid]):.6f}")
        print(f"  Y 範圍: {np.nanmin(Y[valid]):.6f} ~ {np.nanmax(Y[valid]):.6f}")
    else:
        print("  [WARN] 無有效深度點，跳過此 view")
        continue

    pts = np.stack([X[valid], Y[valid], Z[valid]], axis=1)  # (N,3)
    cols = rgb[valid]  # (N,3) uint8

    # ---- 等比縮放到 target_size
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    extent = float(np.max(maxs - mins))
    if extent > 0:
        scale = target_size / extent
        pts *= scale
    else:
        scale = 1.0
    print(f"  點數: {pts.shape[0]} | 最大邊: {extent:.4f} → 縮放比例: {scale:.4f}")

    # ---- 若為 front view，套用 Z 軸位移（於縮放後）
    # 使用者可用 --front-z-shift 指定偏移量（正值會增加 z）
    front_shift = getattr(args, "front_z_shift", 0.0)
    if view.lower() == "front" and abs(front_shift) > 0.0:
        pts[:, 2] = pts[:, 2] + float(front_shift)
        print(f"  Applied front Z shift: {front_shift:.6f}")

    # ---- 輸出 PLY
    ply_out = folder / f"{view}_rgbd_pointcloud.ply"
    save_ply_with_color(ply_out, pts, cols)
    print(f"  已輸出：{ply_out.resolve()}")
    processed += 1

print(f"\n[DONE] 完成 {processed} 個 view。輸出位置：{folder.resolve()}")
