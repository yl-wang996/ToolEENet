import numpy as np
import open3d as o3d
from tf.transformations import compose_matrix, decompose_matrix
import tf


def sample_data(data, num_sample):
	""" data is in N x ...
		we want to keep num_samplexC of them.
		if N > num_sample, we will randomly keep num_sample of them.
		if N < num_sample, we will randomly duplicate samples.
	"""
	try:
		N = data.shape[0]
		if (N == num_sample):
			return data, range(N)
		elif (N > num_sample):
			sample = np.random.choice(N, num_sample)
			return data[sample, ...], sample
		else:
			# print(N)
			sample = np.random.choice(N, num_sample - N)
			dup_data = data[sample, ...]
		return np.concatenate([data, dup_data], 0), list(range(N)) + list(sample)
	except Exception as e:
		print(e)

class TfUtils:
	@staticmethod
	def random_tf_M(N=None):
		trans_offset_limit = 0.2
		rnd_Ms = []
		if N is None:
			trans_offset = np.random.uniform(low=-1, high=1, size=(3,)) * trans_offset_limit
			angle_offset = np.random.uniform(low=-1, high=1, size=(3,)) * np.pi
			rnd_M = TfUtils.pose_to_tf_M(
				translate=trans_offset,
				angles=angle_offset,
			)
			return rnd_M
		else:
			for _ in range(N):
				trans_offset = np.random.uniform(low=-1, high=1, size=(3,)) * trans_offset_limit
				angle_offset = np.random.uniform(low=-1, high=1, size=(3,)) * np.pi
				rnd_M = TfUtils.pose_to_tf_M(
					translate=trans_offset,
					angles=angle_offset,
				)
				rnd_Ms.append(np.expand_dims(rnd_M, axis=0))
			rnd_Ms = np.concatenate(rnd_Ms, axis=0)
			return rnd_Ms



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
		# angles here is radians
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

	@staticmethod
	def random_transform(points, ee_poses:list, obj_pose, rnd_M=None):
		obj_trans = obj_pose[:3, 3].T
		if rnd_M is None:
			rnd_M = TfUtils.random_tf_M()
		# randomize the points cloud
		points -= obj_trans
		ones = np.expand_dims(np.ones(points.shape[0]), axis=-1)
		points = np.concatenate([points, ones], axis=-1)
		points = points.T
		points = np.dot(rnd_M, points)
		points = points.T[:, :3]
		points += obj_trans
		
		# randomize the ee poses
		ee_poses_copy = ee_poses.copy()
		for idx, ee_pose in enumerate(ee_poses_copy):
			ee_pose[:3, 3] -= obj_pose[:3, 3]
			ee_pose = np.dot(rnd_M, ee_pose)
			ee_pose[:3, 3] += obj_pose[:3, 3]
			ee_poses[idx] = np.asarray(ee_pose)
		
		# randomize the obj pose
		new_obj_pose = obj_pose.copy()
		new_obj_pose[:3, 3] -= obj_pose[:3, 3]
		new_obj_pose = np.dot(rnd_M, new_obj_pose)
		new_obj_pose[:3, 3] += obj_pose[:3, 3]
		
		return np.asarray(points), ee_poses, np.asarray(new_obj_pose)