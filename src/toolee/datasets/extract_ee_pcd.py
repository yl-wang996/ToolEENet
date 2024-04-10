import os
import time

from multiprocessing import Pool
import numpy as np
from utils.file_utils import MetaUtils, get_examples
from configs.config import get_config, get_affordance_id_from_name
from tqdm import tqdm
import open3d as o3d
from utils.data_tools import sample_data
import random


	

def depth_to_pointcloud(depth_buffer, rgb_buffer, seg_buffer, seg_id, camera_proj_matrix, width, height):
	fu = 2 / camera_proj_matrix[0, 0]
	fv = 2 / camera_proj_matrix[1, 1]
	centerU = width / 2
	centerV = height / 2
	
	u = range(0, rgb_buffer.shape[1])
	v = range(0, rgb_buffer.shape[0])
	
	u, v = np.meshgrid(u, v)
	u = u.astype(float)
	v = v.astype(float)
	
	Z = depth_buffer
	X = -(u - centerU) / width * Z * fu
	Y = (v - centerV) / height * Z * fv
	
	Z = Z.flatten()
	depth_valid = Z > -10001
	seg_valid = seg_buffer.flatten() == seg_id
	valid = np.logical_and(depth_valid, seg_valid)
	X = X.flatten()
	Y = Y.flatten()
	
	position = np.vstack((X, Y, Z, np.ones(len(X))))[:, valid].T
	colors = rgb_buffer.reshape((-1, 3))[valid]
	
	points = position[:, 0:3]
	# if points.shape[0] < cfg.num_points:
	# 	print(f"Warning: {points.shape[0]} points < 1024 in the point cloud, may occlusion or other problems")
	return points, colors

def extract_ee_pcd(example):
	meta_name, ee_name = example
	data_root = '/dataSSD/yunlong/dataspace/DatasetToolEE'
	meta_util = MetaUtils(data_root, meta_name)
	cat, obj, idx = meta_util.get_cat_obj_id()
	pcd_file_name = f"ee_{cat}_{obj}_{ee_name}_{idx:04d}.pcd"
	if not os.path.exists(os.path.join(data_root, cat, obj, pcd_file_name)) or meta_util.get(f'ee_pcd_path_{ee_name}') is None:
		if not os.path.exists(os.path.join(data_root, cat, obj, pcd_file_name)):
			depth = meta_util.get_depth_array()
			seg = meta_util.get_affordance_seg()
			rgb = meta_util.get_image()
			height, width = meta_util.get_cam_hw()
			seg_id = get_affordance_id_from_name(cat, ee_name)
			proj_matrix = meta_util.get_cam_proj_matrix()
			points, colors = depth_to_pointcloud(
				depth_buffer=depth,
				rgb_buffer=rgb,
				seg_buffer=seg,
				seg_id=seg_id,
				camera_proj_matrix=proj_matrix,
				height=height,
				width=width
			)
			if points.shape[0] < 50:
				print(f"points shape:{points.shape}, {meta_name} {ee_name}  is not visuable, ignore it.")
				return
			try:
				_, sample_idx = sample_data(points, cfg.num_points)
			except Exception as e:
				print(f"Error in {meta_name} {ee_name}")
				print(e)
				return
			points = points[sample_idx]
			colors = colors[sample_idx]
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
			pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
			o3d.io.write_point_cloud(filename=os.path.join(data_root, cat, obj, pcd_file_name), pointcloud=pcd)

if __name__ == '__main__':
	cfg = get_config()
	data_root = cfg.data_path
	examples_train = get_examples(data_root, "all_training_examples_ee.txt").tolist()
	examples_novel = get_examples(data_root, "novel_examples_ee.txt").tolist()
	
	# with Pool(processes=50) as pool:
	# 	for _ in tqdm(pool.imap_unordered(extract_ee_pcd, examples_train), total=len(examples_train)):
	# 		pass
	#
	# with Pool(processes=60) as pool:
	# 	for _ in tqdm(pool.imap_unordered(extract_ee_pcd, examples_novel), total=len(examples_novel)):
	# 		pass
	#
	#
	# not_vis_objs = [
	# 	'wrench_15',
	# 	'hammer_04',
	# 	'wrench_18',
	# ]
	# not_visible_list = []
	# for example in tqdm(examples_train):
	# 	meta_name, ee_name = example
	# 	for obj in not_vis_objs:
	# 		if obj in meta_name:
	# 			meta_util = MetaUtils(data_root, meta_name)
	# 			if not meta_util.get_ee_pcd_path(ee_name):
	# 				not_visible_list.append(example)
	#
	# for example in tqdm(examples_novel):
	# 	meta_name, ee_name = example
	# 	for obj in not_vis_objs:
	# 		if obj in meta_name:
	# 			meta_util = MetaUtils(data_root, meta_name)
	# 			if not meta_util.get_ee_pcd_path(ee_name):
	# 				not_visible_list.append(example)
	#
	# np.savetxt(os.path.join(data_root, "not_visible_ee.txt"), np.asarray(not_visible_list), fmt='%s')
	
	examples_not_visuable = np.loadtxt(os.path.join(data_root, "not_visible_ee.txt"), dtype=str,delimiter=',').tolist()
	for example in tqdm(examples_not_visuable):
		if example in examples_train:
			examples_train.remove(example)
			continue
		if example in examples_novel:
			examples_novel.remove(example)

	np.savetxt(os.path.join(data_root, "all_training_examples_ee_visible.txt"), np.asarray(examples_train), fmt='%s',delimiter=',')
	np.savetxt(os.path.join(data_root, "novel_examples_ee_visible.txt"), np.asarray(examples_novel), fmt='%s',delimiter=',')

	train_ratio = 0.8
	random.seed(0)
	random.shuffle(examples_train)
	train_num = int(len(examples_train) * train_ratio)
	train_examples = examples_train[:train_num]
	val_examples = examples_train[train_num:]

	np.savetxt(os.path.join(data_root, "train_examples_ee_visible.txt"), np.asarray(train_examples), fmt='%s',delimiter=',')
	np.savetxt(os.path.join(data_root, "val_examples_ee_visible.txt"), np.asarray(val_examples), fmt='%s',delimiter=',')


	
		
		
		
		
	
