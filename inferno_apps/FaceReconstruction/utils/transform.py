import numpy as np
import cv2
R = np.array([[0.8713, 0, -0.4907],
                        [0.1338, 0.9621, 0.2375],
                        [0.4721, -0.2726, 0.8383]])



# R = R.transpose()  # because opencv assumes column-major

rvec, _ = cv2.Rodrigues(R)   # R -> rvec (3x1)
R2, _  = cv2.Rodrigues(rvec) # rvec -> R


print("Original Rotation Matrix R:")
print(R)
print("\nRotation Vector rvec:")
print(rvec)


def euler_to_R(pitch, yaw, roll, order='ZYX', degrees=False):
    # pitch=x, yaw=y, roll=z
    if degrees:
        pitch, yaw, roll = np.deg2rad([pitch, yaw, roll])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]], dtype=np.float64)

    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [0,            1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float64)

    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll),  np.cos(roll), 0],
                   [0,             0,            1]], dtype=np.float64)

    if order == 'ZYX':       # 常見：roll→yaw→pitch
        R = Rz @ Ry @ Rx
    elif order == 'XYZ':     # 有些模型會用這個
        R = Rx @ Ry @ Rz
    else:
        raise ValueError("Unsupported order")
    return R

ans = np.array([[-0.3977, 0.7546, -0.0508]])
pitch, yaw, roll = -0.3977, 0.7546, -0.0508  # 通常是弧度
R_rotation_vector = euler_to_R(pitch, yaw, roll, order='ZYX')
ans_R_rotation_vector = np.array(R_rotation_vector)
ans_array,_ = cv2.Rodrigues(ans)

print("ans array")
print(ans_array)
print("ans_R_rotation_vector")
print(ans_R_rotation_vector)
print("TRUE RRRRR")
print(R)
