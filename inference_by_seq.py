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


def plot_trajectory(positions, show_axes=False):
    """
    Plot the trajectory in 3D space.

    Parameters:
    - positions: Nx3 array of positions.
    - show_axes: If True, plot orientation axes at each pose.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(positions[:,0], positions[:,1], positions[:,2], label='Trajectory', marker='.', color="red")

    # Set labels
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('3D Trajectory Visualization')
    ax.legend()
    ax.grid(True)

    # Equal aspect ratio
    max_range = np.array([positions[:,0].max()-positions[:,0].min(),
                          positions[:,1].max()-positions[:,1].min(),
                          positions[:,2].max()-positions[:,2].min()]).max() / 2.0

    mid_x = (positions[:,0].max()+positions[:,0].min()) * 0.5
    mid_y = (positions[:,1].max()+positions[:,1].min()) * 0.5
    mid_z = (positions[:,2].max()+positions[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


# Example Usage
if __name__ == "__main__":
    obj = load_from_pickle()
    oxts_data, poses = obj

    train_data = oxts_data[0:8008]
    train_poses = poses[0:8008]
    train_dataset = KittiGPSIMUDataset(train_data, train_poses, save_stats=False)
    print("Kitti GPS/IMU Train Dataset Loaded:  ", len(train_data))

    print("train_dataset[0]:", train_dataset[0])

    val_data = oxts_data[8008:19103]
    val_poses = poses[8008:19103]
    val_dataset = KittiGPSIMUDataset(val_data, val_poses, save_stats=False)
    print("Kitti GPS/IMU Val Dataset Loaded:  ", len(val_data))


    # Initialize the model
    model = KKGPSIMULocalizationModel()


    ## Inference Example
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    # Load the best model
    model.load_state_dict(torch.load('best_odometry_model.pth'))
    model.to(device)

    # Load the statistics for inference
    stats_file = "dataset_stats.json"
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    dataset_mean = torch.tensor(stats["dataset_mean"])
    dataset_std = torch.tensor(stats["dataset_std"])
    pose_mean = np.array(stats["pose_mean"])
    pose_std = np.array(stats["pose_std"])

    ## Inference the 1st seq
    # Retrieve the 1st seq

    gt_pose_list = []
    predicted_pose_list = []

    ## training_seq: [0:8008]
    ## val_seq: [0:11095]

    for i in range(11095):
        #oxts, gt_pose = train_dataset[i]
        oxts, gt_pose = val_dataset[i]
        #print("oxts.shape:", oxts.shape)  #(23,)
        print("gt_pose:", gt_pose)  
        print("gt_pose.numpy():", gt_pose.numpy())
        gt_pose_list.append(gt_pose.numpy())

        # Predict 
        predict_pose = infer(model, oxts, dataset_mean, dataset_std, pose_mean, pose_std, device=device)

        predicted_pose_list.append(predict_pose)
        print("Predicted pose:", predict_pose)
        print("Predicted pose.shape:", predict_pose.shape)


    gt_pose_list_np = np.array(gt_pose_list)
    predicted_pose_list_np = np.array(predicted_pose_list)

    print("=== gt_pose_list ===")
    print(gt_pose_list_np.shape)
    print("====================")
    print("=== predicted_pose_list ===")
    print(predicted_pose_list_np.shape)
    print("===========================")


    np.save('gt_pose_list_np', gt_pose_list_np)
    np.save('predicted_pose_list_np', predicted_pose_list_np)
    #plot_trajectory(np.array(gt_pose_list))


