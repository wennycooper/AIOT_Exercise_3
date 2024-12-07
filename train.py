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

class KKGPSEncoder(nn.Module):
    def __init__(self):
        super(KKGPSEncoder, self).__init__()
        # First stage: Local feature extraction
        self.fc1 = nn.Linear(3, 3)  # 將3維輸入轉換為128維
        #self.fc2 = nn.Linear(128, 512)
        #self.fc3 = nn.Linear(512, 2048)

        # Global feature extraction
        #self.fc4 = nn.Linear(2048, 512)
        #self.fc5 = nn.Linear(512, 128)

        # Pose regression (3-DoF: x, y, z)
        #self.fc6 = nn.Linear(128, 3)

        # initialization 
        nn.init.uniform_(self.fc1.weight)
        #nn.init.uniform_(self.fc2.weight)
        #nn.init.uniform_(self.fc5.weight)
        #nn.init.uniform_(self.fc6.weight)

    def forward(self, x):
        # x shape: [batch_size, 6]
        #batch_size = x.size(0)

        #print("x.size(0):", batch_size)
        #print("x.shape:", x.shape)
        #print("x:", x)
        #print("x.type():", x.type())

        # Local feature extraction
        x = self.fc1(x)  #b, 128
        #x = F.relu(self.fc2(x))  #b, 512 
        #x = F.relu(self.fc3(x))  #b, 2048
        
        # Fully connected layers for regression
        #x = F.relu(self.fc4(x))  #b, 512
        #x = F.relu(self.fc5(x))  #b, 128
        #x = self.fc6(x)          #b, 3
        
        return x

class KKIMUEncoder(nn.Module):
    def __init__(self):
        super(KKIMUEncoder, self).__init__()
        self.HL = 3   # hidden_layer_num
        self.HD = 20  # hidden state dim

        # First stage: BI-LSTM for feature extraction
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.HD,
            num_layers=self.HL,
            batch_first=True,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(20, 3)  # 將20維IMU輸入轉換為3維
        

        # initialization
        nn.init.uniform_(self.fc1.weight)

    def forward(self, x):
        # x shape: [batch_size, 6]
        #batch_size = x.size(0)

        x = x.unsqueeze(-1)   # convert (b,20) to (b, 20, 1)
        #print("input x.shape", x.shape)  #(b, 20, 1)
        #print("IMU x.shape:", x.shape)
        #print("IMU x:", x)
        #print("IMU x.type():", x.type())



        out, (h, c) = self.lstm(x, None)

        ## h[-1] is the last hidden state (b, 20, 1)
        # Local feature extraction
        x = self.fc1(h[-1].squeeze(0))  #b, 3 
        #x = self.fc1(x)
        #print("IMU encoder output.shape", x.shape)
        #print("IMU encoder output", x)  ## 這裡有值沒問題


        return x



# Define the KKGPSIMULocalizationModel
class KKGPSIMULocalizationModel(nn.Module):
    def __init__(self):
        super(KKGPSIMULocalizationModel, self).__init__()
        self.gps_encoder = KKGPSEncoder()  
        self.imu_encoder = KKIMUEncoder()
        self.fc = nn.Linear(3, 3)   # Combining both outputs for pose prediction

        # initialization
        nn.init.uniform_(self.fc.weight)
    
    def forward(self, oxts):
        # Forward pass     oxts [b, 23]
        feat1 = self.gps_encoder(oxts[:,:3])  # Features from gps (b, 3)
        feat2 = self.imu_encoder(oxts[:,3:])  # Features from imu (b, 20) 

        #print("In KKGPSIMULocalizationModel: oxts.shape:", oxts.shape)
        
        # fuse features from both encoders
        #print("feat1:", feat1)
        #print("feat2:", feat2)
        combined_feat = torch.add(feat1, feat2, alpha=0.1) # Shape (batch_size, 3)
        #combined_feat = feat1

        #print("combined_feat:", combined_feat)
        
        # Predict relative pose
        pose = self.fc(combined_feat)   # (batch_size, 3)

        return pose


# Define the Dataset
from tqdm import tqdm

class KittiGPSIMUDataset(Dataset):
    def __init__(self, oxts_data, poses, save_stats=False, stats_file="dataset_stats.json"):
        self.stats_file = stats_file
        self.oxts_data = oxts_data
        self.poses = poses
        #self.oxts_data = oxts_data[:4]  # debug
        #self.poses = poses[:4]          # debug

        # Compute or load dataset statistics
        if save_stats or not os.path.exists(stats_file):
            self.dataset_mean, self.dataset_std = self._compute_dataset_statistics()
            self.pose_mean, self.pose_std = self._compute_pose_statistics()
            self._save_statistics()
        else:
            self._load_statistics()

    def _compute_dataset_statistics(self):
        """compute mean and standard deviation for the oxts in the dataset."""

        #print("self.oxts_data:", self.oxts_data)
        arrays = [i[:23] for i in self.oxts_data]
        stacked = np.vstack(arrays)
        dataset_mean = np.mean(stacked, axis=0)
        dataset_std = np.std(stacked, axis=0)

        print("len(self.oxts_data):", len(self.oxts_data))
        print("dataset_mean:", dataset_mean)
        print("dataset_std:", dataset_std)

        return dataset_mean, dataset_std

    def _compute_pose_statistics(self):
        """compute mean and std for poses in the dataset."""
        #print("self.poses:", self.poses)
        arrays = [i for i in self.poses]
        stacked = np.vstack(arrays)
        pose_mean = np.mean(stacked, axis=0)
        pose_std = np.std(stacked, axis=0)

        print("len(self.poses):", len(self.poses))
        print("pose_mean:", pose_mean)
        print("pose_std:", pose_std)

        return pose_mean, pose_std

    def _save_statistics(self):
        stats = {
            "dataset_mean": self.dataset_mean.tolist(),
            "dataset_std": self.dataset_std.tolist(),
            "pose_mean": self.pose_mean.tolist(),
            "pose_std": self.pose_std.tolist()
        }
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f)
        print(f"Statistics saved to {self.stats_file}.")

    def _load_statistics(self):
        with open(self.stats_file, 'r') as f:
            stats = json.load(f)
        self.dataset_mean = torch.tensor(stats["dataset_mean"])
        self.dataset_std = torch.tensor(stats["dataset_std"])
        self.pose_mean = torch.tensor(stats["pose_mean"])
        self.pose_std = torch.tensor(stats["pose_std"])
        print(f"Statistics loaded from {self.stats_file}.")


    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        oxts = self.oxts_data[idx][:23]  # [:23]  
        pose = self.poses[idx]

        # Convert to float32 tensors
        oxts = torch.from_numpy(oxts).float()
        pose = torch.from_numpy(pose).float()

        #print("before: oxts.type()", oxts.type())

        # Normalize using dataset-wide mean and std
        oxts = (oxts - self.dataset_mean) / self.dataset_std
        oxts = oxts.float()

        # Normalize relative_pose
        pose = (pose - self.pose_mean) / self.pose_std
        pose = pose.float()

        
        return oxts, pose

    

# Inference Function
def infer(model, oxts, dataset_mean, dataset_std, pose_mean, pose_std, device='cuda'):
    """
    Performs inference to predict the transformation between two point clouds.

    Args:
        model (nn.Module): Trained odometry model.
        oxts (numpy.ndarray): oxts input of shape [N1, 23].
        dataset_mean (torch.Tensor): Dataset-wide mean for normalization.
        dataset_std (torch.Tensor): Dataset-wide std deviation for normalization.
        device (str): Device to perform inference on ('cuda' or 'cpu').

    Returns:
        pose (numpy.ndarray): Predicted pose vector of shape [3] [x, y, z].
    """
    model.eval()
    with torch.no_grad():
        # Preprocess point clouds: sample, pad, and normalize
        #pc1 = preprocess_point_cloud(pc1, num_points)  # Sample or pad
        #pc2 = preprocess_point_cloud(pc2, num_points)

        # Apply normalization
        #pc1 = (pc1 - dataset_mean) / dataset_std
        #pc2 = (pc2 - dataset_mean) / dataset_std

        # Move tensors to the appropriate device and add batch dimension
        #pc1 = pc1.unsqueeze(0).to(device)  # [1, num_points, 3]
        #pc2 = pc2.unsqueeze(0).to(device)  # [1, num_points, 3]
        oxts = oxts.unsqueeze(0).to(device)  # [1, 6]
        #print("in infer, oxts.shape:", oxts.shape)

        # Forward pass to get predicted relative pose
        output = model(oxts)  # [1, 3]
        pose = output.squeeze(0).cpu().numpy()  # [3]

        #print("before denormalized pose:", pose)
        # print("pose_mean:", pose_mean)
        # Denormalized
        # denormalized_pose = pose * pose_std + pose_mean
        #print("after denormalized pose:", denormalized_pose)

    return pose


# Training Function
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, device='cuda'):
    """
    Trains the odometry model.
    
    Args:
        model (nn.Module): The odometry model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
        device (str): Device to train on ('cuda' or 'cpu').
    
    Returns:
        None
    """
    model.to(device)
    criterion = nn.MSELoss()
    #weights = [1, 1, 1, 5, 5, 5]  # 強化 roll, pitch, yaw
    #criterion = WeightedMSELoss(weights=weights).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')

    # Initialize loss history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        #for oxts, pose in train_loader:  ## debug
        for oxts, pose in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):

            oxts, pose = oxts.to(device), pose.to(device)
            optimizer.zero_grad()
            #print("oxts.shape=", oxts.shape)
            #print("pose.shape=", pose.shape)
            #quit()  #here
            outputs = model(oxts)

            #print("in train_model, outputs.shape:", outputs.shape)
            #print("in train_model, poses.shape:", pose.shape)
            loss = criterion(outputs, pose)

            #print("oxts inputs:", oxts)
            #print("gt poses:", pose)
            #print("est outputs:", outputs)
            #print("loss.item():", loss.item())
            #print("oxts.size(0):", oxts.size(0))



            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * oxts.size(0)
            #print("running_loss:", running_loss)

            #quit()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_train_loss:.6f}")
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for oxts, pose in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            #for oxts, pose in val_loader:
                oxts, pose = oxts.to(device), pose.to(device)
                outputs = model(oxts)
                loss = criterion(outputs, pose)

                #print("oxts inputs:", oxts)
                #print("gt poses:", pose)
                #print("est outputs:", outputs)
                #print("loss.item():", loss.item())
                #print("oxts.size(0):", oxts.size(0))
                val_loss += loss.item() * oxts.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {epoch_val_loss:.6f}")
        
        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_odometry_model.pth')
            print("Best model saved.")


        # Save the loss history to a JSON file
        save_dir = "./"
        loss_history_path = os.path.join(save_dir, 'loss_history.json')
        with open(loss_history_path, 'w') as f:
            json.dump(history, f, indent=4)

        print(f"Loss history saved to {loss_history_path}.")
    
    print("Training Complete.")




# Example Usage
if __name__ == "__main__":
    obj = load_from_pickle()
    oxts_data, poses = obj

    train_data = oxts_data[0:8008]
    train_poses = poses[0:8008]
    train_dataset = KittiGPSIMUDataset(train_data, train_poses, save_stats=True)
    print("Kitti GPS/IMU Train Dataset Loaded:  ", len(train_data))

    print("train_dataset[0]:", train_dataset[0])

    val_data = oxts_data[8008:19103]
    val_poses = poses[8008:19103]
    val_dataset = KittiGPSIMUDataset(val_data, val_poses, save_stats=False)
    print("Kitti GPS/IMU Val Dataset Loaded:  ", len(val_data))


    batch_size = 64
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    
    # Initialize the model
    model = KKGPSIMULocalizationModel()


    # Load pretrained model
    # model.load_state_dict(torch.load('./best_odometry_model.pth'))

    # DP
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #if torch.cuda.device_count() > 1:
    #    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #    model = nn.DataParallel(model)
    
    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device=device)
    
    # Save the final model
    #torch.save(model.state_dict(), 'final_odometry_model.pth')
    #print("Final model saved.")
    

    ## Inference Example
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    # Initialize the model
    model = KKGPSIMULocalizationModel()
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
    
    ## Inference the 1st example
    # Retrieve the 1st example
    oxts, gt_pose = train_dataset[0]
    print("train_dataset[0]:", train_dataset[0])
    print("gt_pose:", gt_pose)
    # Predict 
    predict_pose = infer(model, oxts, dataset_mean, dataset_std, pose_mean, pose_std, device=device)
    print("Predicted pose:", predict_pose)
    print("Predicted pose.shape:", predict_pose.shape)

    ## Inference the 8007th example
    # Retrieve the 8007th example
    oxts, gt_pose = train_dataset[8007]
    print("train_dataset[8007]:", train_dataset[8007])
    print("gt_pose:", gt_pose)
    # Predict
    predict_pose = infer(model, oxts, dataset_mean, dataset_std, pose_mean, pose_std, device=device)
    print("Predicted pose:", predict_pose)
    print("Predicted pose.shape:", predict_pose.shape)

