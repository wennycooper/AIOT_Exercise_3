import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import json
import pickle
from load_oxts import *
from train import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories(ground_truth, ours, seq_id, show_axes=False):
    """
    Plot two trajectories in 3D space.

    Parameters:
    - ground_truth: Nx3 array of ground truth positions.
    - ours: Nx3 array of estimated positions.
    - show_axes: If True, plot orientation axes at each pose.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth trajectory
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
            label='Ground Truth', marker='o', color="blue", alpha=0.8)

    # Plot our trajectory
    ax.plot(ours[:, 0], ours[:, 1], ours[:, 2],
            label='Ours', marker='^', color="red", alpha=0.8)

    # Set labels
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('3D Trajectories seq:' + str(seq_id))
    ax.legend()
    ax.grid(True)

    # Equal aspect ratio
    all_positions = np.vstack((ground_truth, ours))
    max_range = np.array([all_positions[:, 0].max() - all_positions[:, 0].min(),
                          all_positions[:, 1].max() - all_positions[:, 1].min(),
                          all_positions[:, 2].max() - all_positions[:, 2].min()]).max() / 2.0

    mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
    mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
    mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

# Example usage:
# ground_truth = np.random.rand(10, 3)  # Replace with actual data
# ours = np.random.rand(10, 3)          # Replace with actual data
# plot_trajectories(ground_truth, ours)


def compute_translation_errors(gt_pose, predicted_pose):
    """
    計算絕對平移誤差 (m) 和百分比平移誤差 (%)

    :param gt_pose: Ground truth poses, shape (N, 3)
    :param predicted_pose: Predicted poses, shape (N, 3)
    :return: Absolute translation error (m), Translation error (%)
    """
    # 確保輸入 shape 相同
    assert gt_pose.shape == predicted_pose.shape, "Ground truth and predicted poses must have the same shape"

    # 計算每個點的絕對平移誤差
    translation_error = np.linalg.norm(gt_pose - predicted_pose, axis=1)  # Shape: (N,)

    # 計算絕對平移誤差 (m)
    ate = np.mean(translation_error)

    # 計算 ground truth 每個點的模長（距離原點的距離）
    gt_distances = np.linalg.norm(gt_pose, axis=1)  # Shape: (N,)

    # 避免分母為 0
    gt_distances[gt_distances == 0] = 1e-6

    # 計算每個點的百分比誤差
    percent_error = (translation_error / gt_distances) * 100  # Shape: (N,)

    # 平均百分比誤差
    mean_percent_error = np.mean(percent_error)

    return ate, mean_percent_error

# 測試數據
ground_truth = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
predicted_pose = np.array([[0, 0, 0], [1.1, 1.0, 0.9], [2.2, 2.1, 1.8]])

ate, percent_error = compute_translation_errors(ground_truth, predicted_pose)

print(f"Absolute Translation Error (m): {ate:.4f}")
print(f"Translation Error (%): {percent_error:.2f}%")



# Example Usage
if __name__ == "__main__":

    gt_pose_list_np = np.load('gt_pose_list_np.npy')
    predicted_pose_list_np = np.load('predicted_pose_list_np.npy')


    # validation set num of entries by sequence
    A = [465, 147, 243, 257, 421, 809, 114, 215, 165, 349, 1176, 774, 694, 152, 850, 701, 510, 305, 180, 404, 173, 203, 436, 430, 316, 176, 170, 85, 175]


    idx_begin = 0
    idx_end = 0
    for i, count in enumerate(A):
        idx_end += count
        ate, percent_error = compute_translation_errors(gt_pose_list_np[idx_begin:idx_end], predicted_pose_list_np[idx_begin:idx_end])
        print(f"Seq:", i)
        print(f"Absolute Translation Error (m): {ate:.4f}")
        print(f"Translation Error (%): {percent_error:.2f}%")
        print(f"===========================================")
        plot_trajectories(
                gt_pose_list_np[idx_begin:idx_end],
                predicted_pose_list_np[idx_begin:idx_end],
                i)

        idx_begin = idx_end

