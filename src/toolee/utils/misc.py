import copy
import os
import sys

import numpy as np
import torch

sys.path.append('..')

from scipy.spatial.transform import Rotation as R
from utils.transforms import rotation_6d_to_matrix

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True
    
def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        # print(N)
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)
   
def get_rot_matrix(batch_pose):
    """
        'rot_matrix' -> batch_pose [B, 6]
        
    Return: rot_matrix [B, 3, 3]
    """
    
    rot_mat = rotation_6d_to_matrix(batch_pose).permute(0, 2, 1)
    return rot_mat

def transform_batch_pts(batch_pts, batch_pose, pose_mode='rot_matrix', inverse_pose=False):
    """
    Args:
        batch_pts [B, N, C], N is the number of points, and C [x, y, z, ...]
        batch_pose [B, C], [quat/rot_mat/euler, trans]
        pose_mode is from ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'rot_matrix']
        if inverse_pose is true, the transformation will be inversed
    Returns:
        new_pts [B, N, C]
    """
    assert pose_mode in ['rot_matrix', 'rot_matrix_symtr'], f"the rotation mode {pose_mode} is not supported!"
    B = batch_pts.shape[0]
    rot = batch_pose[:, :6]
    loc = batch_pose[:, 6:9]


    rot_mat = get_rot_matrix(rot)
    if inverse_pose == True:
        rot_mat, loc = inverse_RT(rot_mat, loc)
    loc = loc[..., np.newaxis]    
    
    trans_mat = torch.cat((rot_mat, loc), dim=2)
    trans_mat = torch.cat((trans_mat, torch.tile(torch.tensor([[0, 0, 0, 1]]).to(trans_mat.device), (B, 1, 1))), dim=1)
    
    new_pts = copy.deepcopy(batch_pts)
    padding = torch.ones([batch_pts.shape[0], batch_pts.shape[1], 1]).to(batch_pts.device)
    pts = torch.cat((batch_pts[:, :, :3], padding), dim=2) 
    new_pts[:, :, :3] = torch.matmul(trans_mat.to(torch.float32), pts.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]
    
    return new_pts
  
def inverse_RT(batch_rot_mat, batch_trans):
    """
    Args: 
        batch_rot_mat [B, 3, 3]
        batch_trans [B, 3]
    Return:
        inversed_rot_mat [B, 3, 3]
        inversed_trans [B, 3]       
    """
    trans = batch_trans[..., np.newaxis]
    inversed_rot_mat = batch_rot_mat.permute(0, 2, 1)
    inversed_trans = - inversed_rot_mat @ trans
    return inversed_rot_mat, inversed_trans.squeeze(-1)

""" https://arc.aiaa.org/doi/abs/10.2514/1.28949 """
""" https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions """
""" http://tbirdal.blogspot.com/2019/10/i-allocate-this-post-to-providing.html """

def average_quaternion_batch(Q, weights=None):
    """calculate the average quaternion of the multiple quaternions
    Args:
        Q (tensor): [B, num_quaternions, 4]
        weights (tensor, optional): [B, num_quaternions]. Defaults to None.

    Returns:
        oriented_q_avg: average quaternion, [B, 4]
    """
    
    if weights is None:
        weights = torch.ones((Q.shape[0], Q.shape[1]), device=Q.device) / Q.shape[1]
    A = torch.zeros((Q.shape[0], 4, 4), device=Q.device)
    weight_sum = torch.sum(weights, axis=-1)

    oriented_Q = ((Q[:, :, 0:1] > 0).float() - 0.5) * 2 * Q
    A = torch.einsum("abi,abk->abik", (oriented_Q, oriented_Q))
    A = torch.sum(torch.einsum("abij,ab->abij", (A, weights)), 1)
    A /= weight_sum.reshape(A.shape[0], -1).unsqueeze(-1).repeat(1, 4, 4)

    q_avg = torch.linalg.eigh(A)[1][:, :, -1]
    oriented_q_avg = ((q_avg[:, 0:1] > 0).float() - 0.5) * 2 * q_avg
    return oriented_q_avg


def average_quaternion_numpy(Q, W=None):
    if W is not None:
        Q *= W[:, None]
    eigvals, eigvecs = np.linalg.eig(Q.T@Q)
    return eigvecs[:, eigvals.argmax()]


def normalize_rotation(rotation, rotation_mode):

    if rotation_mode == 'rot_matrix' or rotation_mode == 'rot_matrix_symtr':
        rot_matrix = get_rot_matrix(rotation)
        rotation[:, :3] = rot_matrix[:, :, 0]
        rotation[:, 3:6] = rot_matrix[:, :, 1]
    else:
        raise NotImplementedError
    return rotation

    
if __name__ == '__main__':
    quat = torch.randn(2, 3, 4)
    quat = quat / torch.linalg.norm(quat, axis=-1).unsqueeze(-1)
    quat = average_quaternion_batch(quat)
    

    