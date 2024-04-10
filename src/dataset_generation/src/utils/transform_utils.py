import numpy as np
import tf
from isaacgym import gymapi
from tf.transformations import compose_matrix, decompose_matrix


class GymUtil:
    @staticmethod
    def quaternion_to_xyz(quaternion):
        if type(quaternion) == np.ndarray:
            quaternion = GymUtil.arrary_to_Quat(quaternion)
        z, y, x = quaternion.to_euler_zyx()
        angle = np.array([x,y,z])
        return angle

    @staticmethod
    def xyz_to_quaternion(x, y, z):
        quat = gymapi.Quat.from_euler_zyx(z, y, x)
        quat = GymUtil.Quat_to_arrary(quat)
        return quat

    @staticmethod
    def degree_to_radian(x, y, z):
       return x / 180 * np.pi, y / 180 * np.pi, z / 180 * np.pi
    @staticmethod
    def Quat_to_arrary(Quat):
        quat = np.zeros(4)
        quat[0] = Quat.x
        quat[1] = Quat.y
        quat[2] = Quat.z
        quat[3] = Quat.w
        return quat
    @staticmethod
    def Vec3_to_arrary(Vec3):
        p = np.zeros(3)
        p[0] = Vec3.x
        p[1] = Vec3.y
        p[2] = Vec3.z
        return p

    @staticmethod
    def array_to_Vec3(arr):
        Vec3 = gymapi.Vec3(3)
        Vec3.x = arr[0]
        Vec3.y = arr[1]
        Vec3.z = arr[2]
        return Vec3
    @staticmethod
    def arrary_to_Quat(arr):
        Quat = gymapi.Quat()
        Quat.x = arr[0]
        Quat.y = arr[1]
        Quat.z = arr[2]
        Quat.w = arr[3]
        return Quat
    @staticmethod
    def transform_to_pose(T:gymapi.Transform):
        p = GymUtil.Vec3_to_arrary(T.p)
        quat = GymUtil.Quat_to_arrary(T.r)
        return p, quat

    @staticmethod
    def pose_to_transform(p, quat):
        if type(p) is np.ndarray:
            p = GymUtil.array_to_Vec3(p)
        if type(quat) is np.ndarray:
            quat= GymUtil.arrary_to_Quat(quat)
        T = gymapi.Transform()
        T.p = p
        T.r = quat
        return T

    @staticmethod
    def tf_M_to_gym_Tranform(M):
        translate, quat = TfUtils.tf_M_to_pose(M)
        return GymUtil.pose_to_transform(
            p=translate,
            quat=quat
        )

class TfUtils:
    @staticmethod
    def gym_Transform_to_tf_M(T):
        p, quat = GymUtil.transform_to_pose(T)
        M = TfUtils.pose_to_tf_M(translate=p, quat=quat)
        return M

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
        # angles here is radius
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
        return tf.transformations.quaternion_from_euler(angles[0], angles[1], angles[2],axes='sxyz')

    @staticmethod
    def quaternion_to_anglexyz(quaternion):
        return tf.transformations.euler_from_quaternion(quaternion,axes='sxyz')

    @staticmethod
    def decompose_tf_M(M):
        scale, shear, angles, trans, persp = decompose_matrix(M)
        quat = TfUtils.anglexyz_to_quaternion(angles)
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
    def tf_M_to_anglexyz(tf_M):
        return tf.transformations.euler_from_matrix(tf_M, axes="sxyz")



if __name__ == '__main__':
    t = [1, 2, 3]
    quat = TfUtils.anglexyz_to_quaternion([0, 0, 0])
    M = TfUtils.pose_to_tf_M(
        translate=t,
        quat=quat
    )
    T = GymUtil.tf_M_to_gym_Tranform(M)

    trans2, quat2 = GymUtil.transform_to_pose(T)
    angles = GymUtil.quaternion_to_xyz(quat2)
    print(trans2, angles)