import os
import time

import open3d as o3d
import numpy as np
from PIL import Image as Img
import yaml

def get_examples(data_root, meta_file_name='all_examples_ee.txt'):
	examples = np.loadtxt(os.path.join(data_root, meta_file_name), dtype=str, delimiter=',')
	return examples

class MetaUtils():
	def __init__(self, data_root, meta_name):
		self.data_root = data_root
		self.meta_file_path = os.path.join(data_root, meta_name)
		self.pose_path = self.meta_file_path.replace('meta', 'pose')
		with open(self.meta_file_path, 'r') as f:
			self.meta_data = yaml.load(f, Loader=yaml.FullLoader)
			f.close()
			time.sleep(0.01)
		self.pose_data = None
		self.hand_pcd_path = self.meta_data['hand_pcd_path']
		self.obj_pcd_path = self.meta_data['obj_pcd_path']
		self.depth_path = self.meta_data['depth_path']
		self.image_path = self.meta_data['image_path']
		self.seg_path = self.meta_data['seg_path']
		self.cat, self.obj, self.idx = self.get_cat_obj_id()
		
	def set(self, key, value):
		self.meta_data[key] = value
	
	def get(self, key):
		if key not in self.meta_data:
			return None
		return self.meta_data[key]
	
	def get_ee_points(self):
		self.load_pose_data()
		return  self.pose_data['ee_point']
	
	def get_img_path(self):
		return os.path.join(self.data_root, self.meta_data['image_path'])
	
	def save_meta(self, new_path=None):
		if new_path is None:
			new_path = self.meta_file_path
		with open(new_path, 'w') as f:
			yaml.dump(self.meta_data, f)
			f.close()
			time.sleep(0.01)

	def get_cat_obj_id(self):
		name = self.meta_file_path.split('/')[-1]
		name = name.replace('meta_', '')
		name = name.replace('.yaml', '')
		id = int(name.split('_')[-1])
		name = name.replace(f'_{id:04d}', '')
		obj = f"{name.split('_')[-2]}_{name.split('_')[-1]}"
		name = name.replace(f'_{obj}', '')
		cat = name
		return cat, obj, id
	
	# get the point cloud of single end effector
	def get_ee_pcd(self, ee_name):
		"""
		Get the point cloud of single end effector
		Args:
			ee_name: the name of ee, e.g. 'grip', 'head1', 'head2'

		Returns:

		"""
		ee_pcd_path = self.get_ee_pcd_path(ee_name)
		if ee_pcd_path is None:
			return None, None
		ee_pcd = o3d.io.read_point_cloud(os.path.join(self.data_root, ee_pcd_path))
		return np.asarray(ee_pcd.points), np.asarray(ee_pcd.colors)
	
	# get the content path of ee points cloud file
	def get_ee_pcd_path(self, ee_name):
		pcd_path = os.path.join(self.cat, self.obj, f"ee_{self.cat}_{self.obj}_{ee_name}_{self.idx:04d}.pcd")
		if not os.path.exists(os.path.join(self.data_root, pcd_path)):
			return None
		return pcd_path
	
	# get the depth array
	def get_depth_array(self):
		depth_path = f"{self.meta_data['depth_path']}"
		if not depth_path.endswith('.npz'):
			depth_path = f"{depth_path}.npz"
		depth_path = os.path.join(self.data_root, depth_path)
		depth = np.load(depth_path)['arr_0']  # -10001 denote invalid depth, which is background or empty space
		return depth
	
	# get the image, rgb
	def get_image(self):
		image_path = os.path.join(self.data_root, self.meta_data['image_path'])
		image = np.asarray(Img.open(image_path))
		return image
	
	# get the hand segmentation id, in the segmentation image(only include background, hand, object), not the affordance segmentation
	def get_hand_seg_id(self):
		return self.meta_data['hand']['seg_id']
	
	# get the obj segmentation id, in the segmentation image(only include background, hand, object), not the affordance segmentation
	def get_obj_seg_id(self):
		return self.meta_data['object']['seg_id']
	
	# get the scale of the object based on the normalized 3D object assets
	def get_obj_scale(self):
		return self.meta_data['object']['scale']
	
	
	# get the point cloud of the object
	def get_obj_point_cloud(self):
		obj_pcd_path = os.path.join(self.data_root, self.obj_pcd_path)
		obj_pcd = o3d.io.read_point_cloud(obj_pcd_path)
		return np.asarray(obj_pcd.points), np.asarray(obj_pcd.colors)
	
	# get the point cloud of the hand, should not been used
	# def get_hand_point_cloud(self):
	# 	print("Warning: the hand point cloud should not been used, it is not accurate.")
	# 	hand_pcd_path = os.path.join(self.data_root, self.meta_data['hand_pcd_path'])
	# 	hand_pcd = o3d.io.read_point_cloud(hand_pcd_path)
	# 	return np.asarray(hand_pcd.points), np.asarray(hand_pcd.colors)
	
	# get the camera height and width
	def get_cam_hw(self):
		return int(self.meta_data['camera']['height']), int(self.meta_data['camera']['width'])
	
	# get the camera projection matrix, intrinsic matrix
	def get_cam_proj_matrix(self):
		return np.asarray(self.meta_data['camera']['projection_matrix'])
	
	# get the camera view matrix, not the extrinsic matrix, it is used for only transform the point cloud between camera frame and world frame.
	def get_cam_view_matrix(self):
		return np.asarray(self.meta_data['camera']['view_matrix'])
	
	# get the camera transform matrix, extrinsic matrix
	def get_cam_tranform(self):
		return np.asarray(self.meta_data['camera']['transform'])
	
	# get the environment base position,  based on the origin of the world frame
	def get_env_base(self):
		return np.asarray(self.meta_data['env_base'])
	
	# get the pose data of the object, the pose is the transformation matrix in camera frame
	def load_pose_data(self):
		if self.pose_data is None:
			with open(self.pose_path, 'r') as f:
				self.pose_data = yaml.load(f, Loader=yaml.FullLoader)
	
	# get the pose of the object, the pose is the transformation matrix in camera frame
	def get_obj_pose(self):
		self.load_pose_data()
		return np.asarray(self.pose_data['obj_pose'])
	
	# get the pose of the end effector, the pose is the transformation matrix in camera frame, return dict{ee_name: pose}
	def get_ee_poses(self, ee_name=None):
		self.load_pose_data()
		if ee_name is None:
			return self.pose_data['ee_pose']
		else:
			return np.asarray(self.pose_data['ee_pose'][ee_name])
	
	# get the names of the end effectors based on this meta file
	def get_ee_names(self):
		self.load_pose_data()
		return list(self.pose_data['ee_pose'].keys())
	
	def get_seg(self):
		seg_path = os.path.join(self.data_root, self.meta_data['seg_path'])
		seg = np.asarray(Img.open(seg_path))
		return seg
	
	# get the segmentation image of the affordance, include background, hand, object, and each of the end effectors
	def get_affordance_seg(self):
		seg_path = os.path.join(self.data_root, self.meta_data['affordance_seg_path'])
		seg = np.asarray(Img.open(seg_path))
		return seg

if __name__ == '__main__':
	data_root = '/dataSSD/yunlong/dataspace/DatasetToolEE'
	asset_root = '/dataSSD/yunlong/dataspace/Dataset3DModel'
	
	with open(os.path.join(data_root, 'all_examples.txt'), 'r') as f:
		meta_names = f.read().splitlines()
		
	meta_name = meta_names[44999]  # 'hammer_grip/hammer_01/meta_hammer_grip_hammer_01_0000.yaml'
	meta = MetaUtils(data_root, meta_name)
	print(meta.get_cat_obj_id())
	print(meta.get_depth_array())
	print(meta.get_image())
	print(meta.get_hand_seg_id())
	print(meta.get_obj_seg_id())
	print(meta.get_obj_scale())
	print(meta.get_obj_point_cloud()[0].shape)
	print(meta.get_cam_hw())
	print(meta.get_cam_proj_matrix())
	print(meta.get_cam_view_matrix())
	print(meta.get_cam_tranform())
	print(meta.get_env_base())
	print(meta.get_obj_pose())
	print(meta.get_ee_poses())