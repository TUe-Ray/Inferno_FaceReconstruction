
import numpy as np

def rodrigues_from_R(R):
    # 和先前給你的穩健版類似，這裡用簡化版
    R = np.asarray(R, float)
    # re-orthonormalize
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    # angle
    cos_t = np.clip((np.trace(R)-1)/2, -1, 1)
    theta = np.arccos(cos_t)
    if np.isclose(theta, 0, atol=1e-8):
        r = 0.5*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
        return r
    ax = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*np.sin(theta))
    ax = ax / (np.linalg.norm(ax)+1e-12)
    return ax*theta

def euler_from_R(R, order="XYZ"):
    # 回傳 radians；定義：R = Rz*Ry*Rx 對應 order="ZYX" 這一類慣例
    R = np.asarray(R, float)
    if order == "XYZ":
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy < 1e-8:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0.0
        else:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        return np.array([x,y,z])
    if order == "ZYX":
        sy = np.sqrt(R[2,2]**2 + R[1,2]**2)
        if sy < 1e-8:
            z = np.arctan2(-R[1,0], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            x = 0.0
        else:
            z = np.arctan2(R[1,0], R[0,0])
            y = np.arctan2(-R[2,0], sy)
            x = np.arctan2(R[2,1], R[2,2])
        return np.array([x,y,z])
    raise ValueError("order not implemented")

R_input = np.array([[0.8713, 0, -0.4907],
                        [0.1338, 0.9621, 0.2375],
                        [0.4721, -0.2726, 0.8383]])

model_triplet = np.array([-0.3977,  0.7546, -0.0508])  # 你看到模型顯示的三個值
scaling_factor = 224/525.0  # 根據你的 crop size 和 scale 推算
delta_x0 = None
delta_y0 = None
delta_pose = [delta_x0, delta_y0]

Fs = {
    "I": np.diag([1,1,1]),
    "Fy": np.diag([1,-1,1]),
    "Fz": np.diag([1,1,-1]),
    "FyFz": np.diag([1,-1,-1]),
}

def try_all(R):
    results = []
    for nameF, F in Fs.items():
        for which in ["R", "R_T"]:
            R_use = R if which=="R" else R.T
            # 前乘、後乘兩種情境：座標換系 vs 物體幀換系
            for mode, Rm in [("F*R*F", F @ R_use @ F),
                             ("F*R",   F @ R_use),
                             ("R*F",   R_use @ F)]:
                rvec = rodrigues_from_R(Rm)
                e_xyz = euler_from_R(Rm, "XYZ")
                e_zyx = euler_from_R(Rm, "ZYX")
                # 和模型三元組做三種對比：Rodrigues、Euler(XYZ)、Euler(ZYX)
                for rep, v in [("rodrigues", rvec), ("euler_XYZ", e_xyz), ("euler_ZYX", e_zyx)]:
                    # 也考慮整體取負（R 和 R.T 的軸角會差一個負號）
                    for sign in [1, -1]:
                        diff = np.linalg.norm(sign*v - model_triplet)
                        results.append((diff, which, nameF, mode, rep, sign, sign*v))
    results.sort(key=lambda x: x[0])
    return results[:10]  # 前 10 名

import numpy as np

def intrinsics_after_crop(K, old_size, center, crop_size=224, scale=1.25):
    size = float(old_size) * float(scale)
    s = float(crop_size) / size
    x0 = float(center[0]) - size/2.0
    y0 = float(center[1]) - size/2.0
    A = np.array([[ s, 0., -s*x0],
                  [ 0., s, -s*y0],
                  [ 0., 0.,    1. ]], dtype=float)
    Kp = A @ K
    return Kp / Kp[2,2]

# 你的數值
K = np.array([[713.2867, 0.      , 512.0],
              [0.      , 705.0691, 384.0],
              [0.      , 0.      ,   1.0]], dtype=float)

Kp = intrinsics_after_crop(K, old_size=420, center=[575.0, 293.5], crop_size=224, scale=1.25)
print(Kp)
# [[304.33565867   0.          85.12      ]
#  [  0.         300.82948267 150.61333333]
#  [  0.           0.           1.        ]]

# 若需要把 R 轉到模型用的 3D 座標：
def flip_R_yz(R):
    F = np.diag([1,-1,-1])
    return F @ R @ F




top = try_all(R_input)
for d, which, Fname, mode, rep, sign, vec in top:
    print(f"err={d:.4f} | {which}, {Fname}, {mode}, {rep}, sign={sign} -> {vec}")
