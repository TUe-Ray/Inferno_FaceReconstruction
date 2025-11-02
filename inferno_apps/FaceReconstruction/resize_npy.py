import numpy as np
import cv2
import os
import json
import shutil
import sys

def to_uint8_for_visual(arr):
    """
    為了存疊圖 JPG：將任意 dtype 的單通道陣列做 min-max normalize 成 uint8。
    仅用於視覺化/存圖，不回寫到原陣列。
    """
    if arr.dtype == np.uint8:
        return arr
    a_min = float(np.min(arr))
    a_max = float(np.max(arr))
    if a_max == a_min:
        return np.zeros_like(arr, dtype=np.uint8)
    vis = cv2.normalize(arr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return vis.astype(np.uint8)

def resize_back_to_old_size(image, old_size, center, fullresolution):
    """
    Resize image to old_size, then paste it into fullresolution at center position (grayscale), keeping old_size.
    image 可以是 uint8 或其他 dtype 的單/三通道陣列。
    """
    # 若是 3 通道先轉灰階；若是浮點也可直接交給 cv2.resize
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Set 0s to np.nan (background to nan)
    image = np.where(image == 0, np.nan, image)

    # OpenCV 要求連續記憶體
    image = np.ascontiguousarray(image)

    # 以雙線性插值縮放為 old_size x old_size（保持原 dtype）
    resized_img = cv2.resize(image, (old_size, old_size), interpolation=cv2.INTER_LINEAR)

      # After resizing, set nan back to 0
    resized_img = np.where(np.isnan(resized_img), 0, resized_img)

    # fullresolution 清零（保持輸入 dtype）
    fullresolution[:] = 0

    x, y = int(center[0]), int(center[1])
    half_size = old_size // 2

    top = y - half_size
    bottom = y + half_size
    left = x - half_size
    right = x + half_size

    fr_top = max(top, 0)
    fr_bottom = min(bottom, fullresolution.shape[0])
    fr_left = max(left, 0)
    fr_right = min(right, fullresolution.shape[1])

    img_top = fr_top - top if fr_top > top else 0
    img_bottom = img_top + (fr_bottom - fr_top)
    img_left = fr_left - left if fr_left > left else 0
    img_right = img_left + (fr_right - fr_left)

    fullresolution[fr_top:fr_bottom, fr_left:fr_right] = resized_img[img_top:img_bottom, img_left:img_right]

    
    # After pasting, set nan in fullresolution back to 0
    fullresolution = np.where(np.isnan(fullresolution), 0, fullresolution)

    return resized_img, fullresolution

def save_blended_image(fullresolution_any_dtype, original_image, save_path, alpha=0.5):
    """
    將 fullresolution 與 original 疊圖後輸出 JPG。
    若 fullresolution 非 uint8，會做 min-max normalize 僅用於可視化。
    """
    # 先確保是 2D 單通道
    if fullresolution_any_dtype.ndim == 3:
        fullresolution_any_dtype = cv2.cvtColor(fullresolution_any_dtype, cv2.COLOR_BGR2GRAY)

    full_vis = to_uint8_for_visual(fullresolution_any_dtype)

    if full_vis.ndim == 2:
        full_color = cv2.cvtColor(full_vis, cv2.COLOR_GRAY2BGR)
    else:
        full_color = full_vis

    if original_image.ndim == 2:
        original_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        original_color = original_image

    if full_color.shape != original_color.shape:
        original_color = cv2.resize(original_color, (full_color.shape[1], full_color.shape[0]))

    blended = cv2.addWeighted(full_color, alpha, original_color, 1 - alpha, 0)
    cv2.imwrite(save_path, blended)
def ensure_single_channel_2d(arr):
    """
    支援 (H,W), (H,W,1), (H,W,3/4), (3/4,H,W) 等情況。
    對 (3,H,W) / (4,H,W) 會自動移軸成 (H,W,3/4) 再轉灰階。
    """
    arr = np.asarray(arr)

    # 已經是 2D
    if arr.ndim == 2:
        return arr

    # (H,W,3) or (H,W,4)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_BGR2GRAY)

    # (3,H,W) or (4,H,W)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        hwc = np.moveaxis(arr, 0, -1)  # -> (H,W,C)
        return cv2.cvtColor(np.ascontiguousarray(hwc), cv2.COLOR_BGR2GRAY)

    # (H,W,1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]

    # 其他情況，先壓成 2D（取第一張）
    while arr.ndim > 2:
        arr = arr[0]
    return arr

def load_depth_any(depth_base_path):
    """
    依序嘗試載入：
      1) .npy  -> 回傳 (array, 'npy')
      2) .json -> 回傳 (array, 'json')
      3) .jpg  -> 回傳 (image array, 'jpg')
    若都不存在，回傳 (None, None)
    """
    npy_path = depth_base_path + '.npy'
    json_path = depth_base_path + '.json'
    jpg_path = depth_base_path + '.jpg'

    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        return arr, 'npy'
    if os.path.exists(json_path):
        with open(json_path, 'r') as jf:
            data = json.load(jf)
        arr = np.array(data)
        return arr, 'json'
    if os.path.exists(jpg_path):
        img = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)
        return img, 'jpg'
    return None, None

if __name__ == '__main__':
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input('Enter the folder path: ').strip()
    # Automatically prepend demo/TestSamples if not already present
    if not folder_path.startswith('demo/TestSamples'):
        folder_path = os.path.join('demo', 'TestSamples', folder_path)
    txt_path = os.path.join(folder_path, 'all_size_center.txt')
    if not os.path.exists(txt_path):
        print(f'Cannot find all_size_center.txt in {folder_path}')
        exit(1)

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    

    # 若來源為 JSON，仍維持彙總 JSON；若來源為 NPY，就不彙總（避免超大 JSON）
    fullresolution_dict = {}

    # results_folder 應在 folder_path 下，名稱為 <last part of the path>_resize_result
    results_folder_name = os.path.basename(folder_path) + '_resize_result'
    results_folder = os.path.join(folder_path, results_folder_name)
    os.makedirs(results_folder, exist_ok=True)
    print(f"Results will be saved in: {results_folder}")
    # 將所有 jpg 和 png 檔案轉換為 jpg，並複製到 results_folder，命名為 original.jpg
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.png')):
            src_path = os.path.join(folder_path, fname)
            dst_path = os.path.join(results_folder, 'original.jpg')
            image = cv2.imread(src_path)
            cv2.imwrite(dst_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"Converted and copied: {src_path} -> {dst_path}")
            break  # 只處理第一個找到的檔案作為 original


    if os.path.isdir(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
        if files:
            sample_image_path = os.path.join(folder_path, files[0])
            sample_image = cv2.imread(sample_image_path, cv2.IMREAD_UNCHANGED)
            fullresolution_shape = sample_image.shape[:2]
        else:
            raise FileNotFoundError("No image found in folder to determine fullresolution_shape.")
    else:
        raise FileNotFoundError(f"{folder_path} is not a valid directory.")


    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            # 例: kai-face-1_CP: old_size: 877, center: [1216.0, 1179.0]
            name_part, rest = line.split(':', 1)
            img_name = name_part.strip()

            old_size_str = rest.split('old_size:')[1].split(',')[0].strip()
            old_size = int(old_size_str)

            center_start = rest.find('center:')
            center_str = rest[center_start:].split('[')[-1].split(']')[0]
            center = [float(x) for x in center_str.split(',')]
        except Exception as e:
            print(f'Error parsing line: {line}\n{e}')
            continue

        # 載入深度資料（npy/json/jpg 任一）
        depth_base = os.path.join(folder_path, 'depth_maps', f'{img_name}_depth')
        depth_data, depth_kind = load_depth_any(depth_base)

        if depth_data is None:
            print(f'No depth file found for: {depth_base} (tried .npy/.json/.jpg)')
            continue

        # 統一為單通道 2D
        depth_data_gray = ensure_single_channel_2d(depth_data)
        print(f"{img_name}: loaded {depth_data.shape} -> gray {depth_data_gray.shape}")

        # 印出處理前 max/min
        pre_min = float(np.min(depth_data_gray))
        pre_max = float(np.max(depth_data_gray))
        print(f"[{img_name}] INPUT({depth_kind}) min={pre_min:.6f}, max={pre_max:.6f}")

        # fullresolution 用與來源相同 dtype（盡量保留數值）
        full_dtype = depth_data_gray.dtype
        fullresolution = np.zeros(fullresolution_shape, dtype=full_dtype)

        resized_img, fullresolution = resize_back_to_old_size(depth_data_gray, old_size, center, fullresolution)

        # 印出處理後 max/min
        post_min = float(np.min(fullresolution))
        post_max = float(np.max(fullresolution))
        print(f"[{img_name}] OUTPUT fullresolution min={post_min:.6f}, max={post_max:.6f}")

        # 儲存可視化的 fullresolution JPG（正規化後）
        full_vis = to_uint8_for_visual(fullresolution)
        jpg_path = os.path.join(results_folder, f'{img_name}_fullresolution_img.jpg')
        cv2.imwrite(jpg_path, full_vis)
        print(f"Saved: {jpg_path}")

        # 依來源型別決定儲存格式
        rot90_fullresolution = np.rot90(fullresolution)

        if depth_kind == 'npy':
            # NPY 輸入 -> 結果存 NPY，非旋轉檔名為 original.npy
            npy_path = os.path.join(results_folder, 'original.npy')
            np.save(npy_path, fullresolution)
            print(f"Saved: {npy_path}")
            npy_rot_path = os.path.join(results_folder, f'{img_name}_rot90_fullresolution_img.npy')
            np.save(npy_rot_path, rot90_fullresolution)
            print(f"Saved: {npy_rot_path}")
        else:
            # JSON 或 JPG 輸入 -> 沿用 JSON 輸出（與原程式一致）
            single_json_path = os.path.join(results_folder, f'{img_name}_fullresolution_img.json')
            with open(single_json_path, 'w') as sjf:
                json.dump(fullresolution.tolist(), sjf)
            print(f"Saved: {single_json_path}")
            rot90_json_path = os.path.join(results_folder, f'{img_name}_rot90_fullresolution_img.json')
            with open(rot90_json_path, 'w') as rjf:
                json.dump(rot90_fullresolution.tolist(), rjf)
            print(f"Saved: {rot90_json_path}")
            # 彙總（只對 JSON/JPG 來源，以免 NPY 巨大陣列塞進單一 JSON）
            fullresolution_dict[img_name] = fullresolution.tolist()

        # 疊合原圖（若存在）
        original_cp_path = None
        for ext in ['.jpg', '.png']:
            potential_path = os.path.join(folder_path, f'{img_name}{ext}')
            if os.path.exists(potential_path):
                original_cp_path = potential_path
            break
        if original_cp_path is None:
            print(f'Original CP image not found for: {img_name}')
            continue
        if os.path.exists(original_cp_path):
            original_cp_img = cv2.imread(original_cp_path, cv2.IMREAD_UNCHANGED)
            #original_cp_img = cv2.rotate(original_cp_img, cv2.ROTATE_90_CLOCKWISE)  # 與 fullresolution 方向一致
            overlap_path = os.path.join(results_folder, f'{img_name}_overlap.jpg')
            save_blended_image(fullresolution, original_cp_img, overlap_path, alpha=0.5)
            print(f"Saved: {overlap_path}")

            # 旋轉後的疊圖
            original_cp_img_rot = cv2.rotate(original_cp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rot_overlap_path = os.path.join(results_folder, f'{img_name}_rot90_overlap.jpg')
            save_blended_image(rot90_fullresolution, original_cp_img_rot, rot_overlap_path, alpha=0.5)
            print(f"Saved: {rot_overlap_path}")
        else:
            print(f'Original CP image not found: {original_cp_path}')

        # 報告貼上的有效區域大小（以視覺化後的遮罩 >0 判斷）
        ys, xs = np.where(to_uint8_for_visual(fullresolution) > 0)
        if ys.size and xs.size:
            h = ys.max() - ys.min() + 1
            w = xs.max() - xs.min() + 1
            print(f"[{img_name}] pasted area in fullresolution (h, w): {(h, w)}")
        else:
            print(f"[{img_name}] pasted area in fullresolution: none")

    # 若有彙總（JSON/JPG 來源），存 all_fullresolution_imgs.json
    if len(fullresolution_dict) > 0:
        json_path = os.path.join(results_folder, 'all_fullresolution_imgs.json')
        with open(json_path, 'w') as jf:
            json.dump(fullresolution_dict, jf)
        print(f"Saved JSON summary with {len(fullresolution_dict)} items to: {json_path}")
        print(f"Saved: {json_path}")
    
    
    # 將結果直接複製到 mergeRGBdepth，整個資料夾存入而非分散檔案
    merge_folder = 'mergeRGBdepth'
    if not os.path.exists(merge_folder):
        os.makedirs(merge_folder)
    dst_folder = os.path.join(merge_folder, os.path.basename(results_folder))
    if os.path.exists(dst_folder):
        # 清空 dst_folder 內容但保留資料夾本身
        for filename in os.listdir(dst_folder):
            file_path = os.path.join(dst_folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(dst_folder)
    # 複製 results_folder 內容到 dst_folder
    for filename in os.listdir(results_folder):
        src_path = os.path.join(results_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
    print(f"Copied entire folder {results_folder} -> {dst_folder}")
