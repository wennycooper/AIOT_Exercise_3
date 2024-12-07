import numpy as np
from pyproj import Proj, transform
from load_oxts import *

# 設定參考原點（第一幀 GPS 點）
#def latlon_to_utm(lat, lon, ref_lat, ref_lon):
def latlon_to_utm(lat, lon):
    wgs84 = Proj(proj="latlong", datum="WGS84")
    utm = Proj(proj="utm", zone=33, datum="WGS84")  # 根據 lat/lon 確定 UTM zone
    #ref_x, ref_y = transform(wgs84, utm, ref_lon, ref_lat)
    x, y = transform(wgs84, utm, lon, lat)
    #return x - ref_x, y - ref_y
    return x, y

# 計算旋轉矩陣
def euler_to_rot_matrix(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx  # ZYX order

# 主程序

# 假設 oxts 文件在 dataset/training/oxts 目錄下
oxts_dir = './dataset/training/oxts'
oxts_data = load_oxts_data(oxts_dir)

oxts_data = oxts_data[154+447:154+447+233]
print("len(oxts_data):", len(oxts_data))

poses = []
ref_lat, ref_lon = None, None
for data in oxts_data:  # 從文件讀取每幀 oxts 數據  
    lat, lon, alt, roll, pitch, yaw = data[:6]
    #if ref_lat is None and ref_lon is None:
    #    ref_lat, ref_lon = lat, lon  # 設定第一幀作為參考點

    #x, y = latlon_to_utm(lat, lon, ref_lat, ref_lon)
    x, y = latlon_to_utm(lat, lon)
    z = alt
    R = euler_to_rot_matrix(roll, pitch, yaw)
    T = np.array([x, y, z])
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = T
    poses.append(pose)


print(len(poses))
# poses 包含所有幀的 4x4 位姿矩陣

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for pose in poses:
    ax.scatter(pose[0, 3], pose[1, 3], pose[2, 3], c='b', marker='o')
plt.show()

