import numpy as np
import tf
from tf.transformations import compose_matrix, decompose_matrix
from utils import transforms
import torch
from utils.misc import average_quaternion_batch

class TfUtils:
    
    @staticmethod
    def random_tf_M():
        trans_offset_limit = 0.2
        trans_offset = np.random.uniform(low=-1, high=1, size=(3,)) * trans_offset_limit
        angle_offset = np.random.uniform(low=-1, high=1, size=(3,)) * np.pi
        rnd_M = TfUtils.pose_to_tf_M(
            translate=trans_offset,
            angles=angle_offset,
        )
        return rnd_M
    
    @staticmethod
    def get_prior_pose(ee_prior_M, obj_M, scale):
        # ee_prior_M: the pose based on the obejct pose
        ee_trans, ee_quat = TfUtils.decompose_tf_M(ee_prior_M)
        ee_M = TfUtils.compose_tf_M(trans=ee_trans*scale, quat=ee_quat)
        ee_M = np.dot(obj_M, ee_M)
        return ee_M
    
    @staticmethod
    def rot_matrix_to_rotation_6d(rot_matrix) -> np.ndarray:
        """
        Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
        by dropping the last row. Note that 6D representation is not unique.
        Args:
            matrix: rotation matrices of size (3, 3)

        Returns:
            6D rotation representation, of size (*, 6)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """
        assert rot_matrix.shape == (3, 3)
        return np.reshape(rot_matrix[:2, :], newshape=(6,))
    
    @staticmethod
    def matrix_to_9d_pose(matrix) -> np.ndarray:
        assert matrix.shape == (4, 4)
        rot_matrix = matrix[:3, :3]
        rot = TfUtils.rot_matrix_to_rotation_6d(rot_matrix)
        trans = matrix[:3, 3]
        return np.concatenate([rot,trans], axis=0)
        

    @staticmethod
    def compose_tf_M(trans, angles=None, quat=None,scale=np.array([1,1,1])):
        # M = compose_matrix(scale, shear, angles, trans, persp)
        # sequence of each transform
        # angles: xyz
        if angles is None:
            angles = TfUtils.quaternion_to_anglexyz(quat)
        M = compose_matrix(
            scale=np.asarray(scale),
            shear=None,
            angles=np.asarray(angles),
            translate=np.asarray(trans),
            perspective=None
        )
        return M

    @staticmethod
    def pose_to_tf_M(translate, angles=None,quat=None):
        # angles here is radian
        assert angles is not None or quat is not None, 'either angle or quat must be provide'
        if angles is None:
            angles = TfUtils.quaternion_to_anglexyz(quat)
        M = compose_matrix(
            scale=None,
            shear=None,
            angles=np.asarray(angles),
            translate=np.asarray(translate),
            perspective=None
        )
        return M

    @staticmethod
    def tf_M_to_pose(M):
        scale, shear, angles, translate, perspective = decompose_matrix(M)
        quat = TfUtils.anglexyz_to_quaternion(angles)
        return translate, quat

    @staticmethod
    def apply_tf_M_to_point(M, point):
        return np.dot(M,np.append(point,1))[:-1]

    @staticmethod
    def anglexyz_to_quaternion(angles):
        # quat: xyzw
        return tf.transformations.quaternion_from_euler(angles[0], angles[1], angles[2],axes='sxyz')

    @staticmethod
    def quaternion_to_anglexyz(quaternion):
        return tf.transformations.euler_from_quaternion(quaternion,axes='sxyz')  # return angles in radian

    @staticmethod
    def decompose_tf_M(M):
        scale, shear, angles, trans, persp = decompose_matrix(M)
        quat = TfUtils.anglexyz_to_quaternion(angles)  # xyzw
        return np.asarray(trans), np.asarray(quat)

    @staticmethod
    def concat_tf_M(matrices):
        M = np.identity(4)
        for i in matrices:
            M = np.dot(M, i)
        return M

    @staticmethod
    def anglexyz_to_tf_M(anglexyz):
        return tf.transformations.euler_matrix(anglexyz[0], anglexyz[1], anglexyz[2], axes="sxyz")

    @staticmethod
    def tf_M_to_anglexyz(tf_M, axes="sxyz"):
        return tf.transformations.euler_from_matrix(tf_M, axes=axes)

    @staticmethod
    def get_avg_sRT(selected_sRT):
        ins_num = selected_sRT.shape[0]
        repeat_num = selected_sRT.shape[1]
        reshaped_selected_sRT = selected_sRT.reshape(ins_num * repeat_num, 4, 4)
        quat_wxyz = transforms.matrix_to_quaternion(torch.from_numpy(reshaped_selected_sRT[:, :3, :3])).cuda()
        quat_wxyz = torch.cat((quat_wxyz, torch.tensor(reshaped_selected_sRT[:, :3, 3]).to(quat_wxyz.device)), dim=-1)
        quat_wxyz = quat_wxyz.reshape(ins_num, repeat_num, -1)
        
        average_pred_pose = torch.zeros((quat_wxyz.shape[0], quat_wxyz.shape[-1])).to(quat_wxyz.device)
        average_pred_pose[:, :4] = average_quaternion_batch(quat_wxyz[:, :, :4])
        average_pred_pose[:, 4:] = torch.mean(quat_wxyz[:, :, 4:], dim=1)
        average_sRT = np.identity(4)[np.newaxis, ...].repeat(ins_num, 0)
        average_sRT[:, :3, :3] = transforms.quaternion_to_matrix(average_pred_pose[:, :4]).cpu().numpy()
        average_sRT[:, :3, 3] = average_pred_pose[:, 4:].cpu().numpy()
        return average_sRT

if __name__ == '__main__':
    t = [1, 2, 3]
    quat = TfUtils.anglexyz_to_quaternion([0, 0, 0])  # xyzw
    M = TfUtils.pose_to_tf_M(
        translate=t,
        quat=quat
    )
