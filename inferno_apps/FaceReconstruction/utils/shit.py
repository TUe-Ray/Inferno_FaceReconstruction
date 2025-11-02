
import numpy as np

def euler_to_R(pitch, yaw, roll):
    cx, cy, cz = np.cos([pitch, yaw, roll])
    sx, sy, sz = np.sin([pitch, yaw, roll])
    R = np.array([
        [cz*cy,  cz*sy*sx - sz*cx,  cz*sy*cx + sz*sx],
        [sz*cy,  sz*sy*sx + cz*cx,  sz*sy*cx - cz*sx],
        [-sy,    cy*sx,             cy*cx]
    ])
    return R

R = euler_to_R(1.57, 1.57, 0.0)
print(R)
a = [0, 0, 1]
R = a *R
print(R)