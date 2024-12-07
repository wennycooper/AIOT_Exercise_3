import os
import numpy as np
from pyproj import Proj, transform
import pickle
from tqdm import tqdm

def latlon_to_utm(lat, lon):
    wgs84 = Proj(proj="latlong", datum="WGS84")
    utm = Proj(proj="utm", zone=33, datum="WGS84")  # 根據 lat/lon 確定 UTM zone
    #ref_x, ref_y = transform(wgs84, utm, ref_lon, ref_lat)
    x, y = transform(wgs84, utm, lon, lat)
    #return x - ref_x, y - ref_y
    return x, y


# 讀取單個 oxts 文件的數據
def read_oxts_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            values = [float(value) for value in line.strip().split()]
            data.append(values)
    return np.array(data)


# 主函數：讀取目錄下的所有 oxts 文件
def load_oxts_data(oxts_dir):
    oxts_files = sorted([
        os.path.join(oxts_dir, f) for f in os.listdir(oxts_dir) if f.endswith('.txt')
    ])
    oxts_data = []
    for file_path in oxts_files:
        data = read_oxts_file(file_path)
        oxts_data.extend(data)  # 將每個文件中的所有數據合併到主列表中
    return oxts_data

def convert_kitti_gps_imu_to_numpy():
    # 假設 oxts 文件在 dataset/training/oxts 目錄下
    training_dir = './dataset/training/oxts'
    training_data = load_oxts_data(training_dir)
    testing_dir = './dataset/testing/oxts'
    testing_data = load_oxts_data(testing_dir)

    #print("training len:", len(training_data))
    #print("testing len:", len(testing_data))

    oxts_data = training_data + testing_data
    #print("oxts len:", len(oxts_data))

    #oxts_data = oxts_data[:5]  # 測試少量data

    poses = []
    for data in tqdm(oxts_data, "computing utm:"):  # 從文件讀取每幀 oxts 數據
        lat, lon, alt, roll, pitch, yaw = data[:6]
        x, y = latlon_to_utm(lat, lon)
        z = alt
        pose = np.array([x,y,z])
        poses.append(pose)


    return oxts_data, poses

def save_to_pickle(obj):
    f = open("dataset.pkl", "wb")
    pickle.dump(obj, f)
    f.close()

def load_from_pickle():
    f = open("dataset.pkl", "rb")
    obj = pickle.load(f)
    f.close()

    return obj

# Example Usage
if __name__ == "__main__":
    oxts_data, poses = convert_kitti_gps_imu_to_numpy()
    save_to_pickle((oxts_data, poses))
    obj = load_from_pickle()

    oxts_data, poses = obj



    # 顯示讀取結果
    print(f"Loaded {len(oxts_data)} oxts files.")
    #for i, data in enumerate(oxts_data[:5]):  # 只顯示前5幀
    #    print(f"Frame {i + 1}: {data}")
    #    print(f"{data.shape}")

    print("len(oxts_data):", len(oxts_data))
    print("len(poses):", len(poses))

    print("oxts_data[0].shape:", oxts_data[0].shape)
    print("poses[0].shape:", poses[0].shape)
