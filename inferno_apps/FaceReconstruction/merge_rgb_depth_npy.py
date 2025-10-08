import sys
from pathlib import Path
import numpy as np
from PIL import Image

# ========= 路徑設定 =========
if len(sys.argv) > 1:
    input_folder = sys.argv[1].strip()
else:
    input_folder = input("請輸入資料夾路徑: ").strip()
folder_path = f"mergeRGBdepth/{input_folder}_resize_result"
folder = Path(folder_path)
rgb_path = folder / "original.jpg"
# 改用 NumPy：優先 original.npy / original.npz，否則掃描資料夾
depth_npy_paths = [
    folder / "original.npy",
    folder / "original.npz",
]
if not any(p.exists() for p in depth_npy_paths):
    npy_npz = list(folder.glob("*.npy")) + list(folder.glob("*.npz"))
    if not npy_npz:
        raise FileNotFoundError("找不到任何深度 NumPy 檔（.npy 或 .npz）")
    depth_npy_paths = [npy_npz[0]]
depth_path = next(p for p in depth_npy_paths if p.exists())

ply_out = folder / "rgbd_pointcloud.ply"

# ========= 載入 RGB =========
rgb_img = Image.open(rgb_path).convert("RGB")
rgb = np.asarray(rgb_img, dtype=np.uint8)  # (H,W,3)
H, W = rgb.shape[:2]
cx = (W - 1) / 2.0
cy = (H - 1) / 2.0

# ========= 從 NumPy 載入深度，並轉成 (H,W) =========
Z_mul = 0.09

def load_depth_from_numpy(path: Path, H: int, W: int) -> np.ndarray:
    """
    支援 .npy 與 .npz：
    - .npy：直接載成 array
    - .npz：優先取 'depth' / 'data' / 'arr_0'；否則取第一個 array
    並嘗試將資料 reshape/旋轉 成 (H,W)
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        npz = np.load(path)
        key_candidates = ["depth", "data", "arr_0"]
        key = next((k for k in key_candidates if k in npz), None)
        if key is None:
            # 若沒有常見 key，就取第一個
            if len(npz.files) == 0:
                raise ValueError(f"{path} 為空的 .npz")
            key = npz.files[0]
        arr = npz[key]
    else:
        raise ValueError(f"不支援的副檔名：{path.suffix}")

    arr = np.asarray(arr)

    # 允許 3D 且最後一維為 1 的情況
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    # 合法情形一：已經是 (H,W)
    if arr.ndim == 2 and arr.shape == (H, W):
        return arr.astype(np.float32)

    # 合法情形二：是 (W,H)，嘗試轉置/旋轉
    if arr.ndim == 2 and arr.shape == (W, H):
        return np.rot90(arr).astype(np.float32)

    # 合法情形三：扁平向量，長度為 H*W
    if arr.ndim == 1 and arr.size == H * W:
        return arr.reshape(H, W).astype(np.float32)

    # 合法情形四：其他 2D 尺寸，嘗試旋轉以匹配
    if arr.ndim == 2:
        arr_rot = np.rot90(arr)
        if arr_rot.shape == (H, W):
            return arr_rot.astype(np.float32)

    raise ValueError(
        f"無法將深度陣列形狀 {arr.shape} 轉成與 RGB 相同的 (H,W)=({H},{W})；請確認檔案或手動處理。"
    )

depth_raw = load_depth_from_numpy(depth_path, H, W).astype(np.float32)

# ========= 自動判斷深度單位並正向化 =========
def auto_depth_to_meters_like(depth):
    d = depth.copy()
    # 去除非正數
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        raise ValueError("深度沒有有效值 (>0)")
    dmin, dmax = float(np.percentile(d[valid], 2)), float(np.percentile(d[valid], 98))

    # 判斷大致範圍
    if dmax <= 1.5:  # 0..1 or meters but tiny
        # 視為 0..1 正規化，映射到 0.3~3m 的通用範圍
        z_min, z_max = 0.3, 3.0
        Z = (d - dmin) / (max(dmax - dmin, 1e-6)) * (z_max - z_min) + z_min
    elif dmax <= 255:  # 8-bit 灰階
        z_min, z_max = 0.3, 3.0
        d01 = np.clip((d - dmin) / (max(dmax - dmin, 1e-6)), 0, 1)
        Z = d01 * (z_max - z_min) + z_min
    elif dmax <= 65535:  # 16-bit 灰階或毫米級
        # 先假設毫米，轉公尺
        Z_mm = d
        # 如果像「視差」（近大遠小），反轉比較合理：兩種都試，選擇更集中的一種
        def z_from_mm(mm, inv=False):
            mmv = np.maximum(mm, 1e-6)
            z = (1.0 / mmv) if inv else mmv
            return z * (0.001 if not inv else 1.0)  # normal: mm->m; inverse: 1/mm -> arbitrary

        Z1 = z_from_mm(Z_mm, inv=False)
        Z2 = z_from_mm(Z_mm, inv=True)
        # 用 robust range 判定哪個更合理（範圍較小者）
        r1 = float(np.percentile(Z1[valid], 98) - np.percentile(Z1[valid], 2))
        r2 = float(np.percentile(Z2[valid], 98) - np.percentile(Z2[valid], 2))
        Z = Z1 if r1 <= r2 else Z2
        # 將 Z 做再正規化到 0.3~3m 範圍內，確保可視
        zmin, zmax = float(np.percentile(Z[valid], 2)), float(np.percentile(Z[valid], 98))
        Z = (Z - zmin) / max(zmax - zmin, 1e-6)
        Z = Z * (3.0 - 0.3) + 0.3
    else:
        # 可能是以公尺為單位但量級很大？以 robust 區間縮放到 0.3~3m
        zmin, zmax = float(np.percentile(d[valid], 2)), float(np.percentile(d[valid], 98))
        Z = (d - zmin) / max(zmax - zmin, 1e-6)
        Z = Z * (3.0 - 0.3) + 0.3

    # 保留有效＋正數
    Z[~valid] = np.nan
    return Z

print(f"原始深度 min: {np.nanmin(depth_raw):.6f}, max: {np.nanmax(depth_raw):.6f}")

Z = auto_depth_to_meters_like(depth_raw)  # (H,W), 近似公尺尺度

print(f"正規化後 Z min: {np.nanmin(Z):.6f}, max: {np.nanmax(Z):.6f}")

# ========= 無需內參的簡化回投影 =========
# 使用 f = max(W,H) 的歸一化近似：X = (x-cx)/f * Z, Y 同理
f = float(max(W, H))
x_scale = 1  # x軸縮放係數，預設1.0，可手動調整
y_scale = 1  # y軸縮放係數，預設1.0，可手動調整
xs = np.arange(W, dtype=np.float32)
ys = np.arange(H, dtype=np.float32)

xx, yy = np.meshgrid(xs, ys)  # (H,W)
print(f"影像尺寸: {(W,H)} | 假設焦距 f={f:.1f} | 光心: {(cx,cy)}")

valid = np.isfinite(Z) & (Z > 0)



X = ((xx - cx) / f) * x_scale
Y = ((yy - cy) / f) * y_scale
# 顯示 X 和 Y 的範圍
if np.any(valid):
    x_min, x_max = np.nanmin(X[valid]), np.nanmax(X[valid])
    y_min, y_max = np.nanmin(Y[valid]), np.nanmax(Y[valid])
    print(f"X 範圍: {x_min:.6f} ~ {x_max:.6f}")
    print(f"Y 範圍: {y_min:.6f} ~ {y_max:.6f}")
else:
    print("無有效點，無法顯示 X 和 Y 的範圍")

# # 新增排除條件：若 (Y>0.15 或 Y<-0.5) 且 Z>0 就移除
# exclude = ((Y > 0.15) | (Y < -0.15)) & (Z > )
# # 最終有效遮罩
# valid = valid & (~exclude)

Z = Z * Z_mul * -1




pts = np.stack([X[valid], Y[valid], Z[valid]], axis=1)  # (N,3)
cols = rgb[valid]  # (N,3) uint8

# ========= 等比自動縮放到合理尺寸 =========
target_size = 1.0
mins = np.nanmin(pts, axis=0)
maxs = np.nanmax(pts, axis=0)
extent = max(maxs - mins)  # 最大邊
if extent > 0:
    scale = target_size / extent
    pts *= scale
else:
    scale = 1.0

print(f"點數: {pts.shape[0]} | 原始範圍最大邊: {extent:.4f} → 縮放比例: {scale:.4f}")

# ========= 存成 PLY（彩色） =========
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

save_ply_with_color(ply_out, pts, cols)
print(f"已輸出：{ply_out.resolve()}")
