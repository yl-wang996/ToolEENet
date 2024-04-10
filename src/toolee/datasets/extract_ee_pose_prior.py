# import os.path
#
# import numpy as np
# from tqdm import tqdm
#
# from utils.file_utils import MetaUtils
# from utils.transform_utils import TfUtils
#
# def extract_ee_pose_prior(meta_file="all_training_examples_ee_visible.txt"):
# 	data_root = '/dataSSD/yunlong/dataspace/DatasetToolEE'
# 	save_path = os.path.join(data_root, 'ee_pose_prior.npy')
# 	meta_file_list = np.loadtxt(f'{data_root}/{meta_file}', dtype=str, delimiter=',')
# 	_ee_prior = {}
# 	ee_prior = {}
# 	if os.path.exists(save_path):
# 		print("Loading existing ee pose prior from file")
# 		ee_prior = np.load(save_path, allow_pickle=True).item()
# 	for meta_file, ee_name in tqdm(meta_file_list, 'Extracting ee pose prior'):
# 		# for meta_file, ee_name in meta_file_list[:100]:
# 		meta_util = MetaUtils(data_root, meta_file)
# 		cat, obj, _ = meta_util.get_cat_obj_id()
#
# 		if cat not in _ee_prior:
# 			_ee_prior[cat] = {}
# 		if cat not in ee_prior:
# 			ee_prior[cat] = {}
#
# 		if obj not in _ee_prior[cat]:
# 			_ee_prior[cat][obj] = {}
# 		if obj not in ee_prior[cat]:
# 			ee_prior[cat][obj] = {}
#
# 		ee_pose_RT = meta_util.get_ee_poses(ee_name=ee_name)
# 		if ee_name not in _ee_prior[cat][obj]:
# 			_ee_prior[cat][obj][ee_name] = []
# 		_ee_prior[cat][obj][ee_name].append(ee_pose_RT)
#
# 	for cat in _ee_prior:
# 		for obj in _ee_prior[cat]:
# 			for ee_name in _ee_prior[cat][obj]:
# 				if ee_name in ee_prior[cat][obj]:
# 					continue
# 				ee_poses = np.array(_ee_prior[cat][obj][ee_name])
# 				ee_poses = np.expand_dims(ee_poses, axis=0)
# 				avg_poses = TfUtils.get_avg_sRT(ee_poses)[0]
# 				ee_prior[cat][obj][ee_name] = avg_poses
# 				print(f'{cat}/{obj}/{ee_name}: {avg_poses}')
# 	print("Saving ee pose prior to file")
# 	np.save(save_path, ee_prior)
#
# def load_pose_prior():
# 	data_root = '/dataSSD/yunlong/dataspace/DatasetToolEE'
# 	ee_prior = np.load(os.path.join(data_root, 'ee_pose_prior.npy'), allow_pickle=True).item()
# 	return ee_prior
#
# if __name__ == '__main__':
# 	extract_ee_pose_prior(meta_file="all_training_examples_ee_visible.txt")
# 	extract_ee_pose_prior(meta_file="novel_examples_ee_visible.txt")
# 	ee_prior = load_pose_prior()
# 	print('done')

import numpy as np
import os
from tqdm import tqdm
