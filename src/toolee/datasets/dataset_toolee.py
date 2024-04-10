import copy
import os
import random
import sys
import yaml

import numpy as np
import torch
import torch.utils.data as data

sys.path.insert(0, '../')

from tqdm import tqdm
from configs.config import get_config
from utils.file_utils import MetaUtils
from utils.transform_utils import TfUtils

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
		sample = np.random.choice(N, num_sample - N)
		dup_data = data[sample, ...]
	return np.concatenate([data, dup_data], 0), list(range(N)) + list(sample)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
	"""
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
	batch_dim = matrix.size()[:-2]
	return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def process_batch(batch_sample,
                  device='cuda',
                  is_pred_symtr=False,
                  ):
	"""
	process the batch data
	:param batch_sample: a batch of data from dataloader
	:param device: device to store the data
	:return: processed batch data
	"""
	points = batch_sample['pcl_in'].to(device)
	colors = batch_sample['colors_in'].to(device)
	gt_rotation = batch_sample['gt_rotation'].to(device)
	gt_translation = batch_sample['gt_translation'].to(device)
	
	processed_sample = {}
	processed_sample['pts'] = points  # [bs, 1024, 3]
	processed_sample['pts_color'] = colors  # haven't been used
	processed_sample['id'] = batch_sample['cat_id'].to(device)  # [bs]
	# processed_sample['path'] = batch_sample['pcd_path']
	rot = matrix_to_rotation_6d(gt_rotation.permute(0, 2, 1)).reshape(gt_rotation.shape[0], -1)
	location = gt_translation  # [bs, 3], the ee location which share the same tf with the points cloud
	if is_pred_symtr:
		symtr = batch_sample['symtr'].to(device)
		processed_sample['symtr'] = symtr  # [bs, 3], the symmetry axis, e.g. [0,0,1] denote the z axis is the symmetry axis
		processed_sample['gt_pose'] = torch.cat([rot.float(), location.float(), symtr.float()], dim=-1)  # [bs, 6 + 3 + 3]
	else:
		processed_sample['gt_pose'] = torch.cat([rot.float(), location.float()], dim=-1)
	
	""" zero center """
	# move the pts to the zero center
	num_pts = processed_sample['pts'].shape[1]
	zero_mean = torch.mean(processed_sample['pts'][:, :, :3], dim=1)
	processed_sample['zero_mean_pts'] = copy.deepcopy(processed_sample['pts'])
	processed_sample['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
	# move the gt_pose to the zero center
	processed_sample['zero_mean_gt_pose'] = copy.deepcopy(processed_sample['gt_pose'])
	processed_sample['zero_mean_gt_pose'][:, 6:9] -= zero_mean
	processed_sample['pts_center'] = zero_mean
	return processed_sample

class TooleeDataset(data.Dataset):
	def __init__(self,
				 mode='train',  # train, test, val, novel
				 data_dir=None,
				 n_pts=1024,
				 per_obj=None,
				 task_type='ee_pose',
				 pred_symtr=False,
				 ):
		self.cat_name_to_id = {
			'hammer_grip': 0,
			'screwdriver': 1,
			'wrench': 2,
		}
		
		self.cat_ee_map = {
			'hammer_grip': ['head1', 'grip'],
			'screwdriver': ['head1'],
			'wrench': ['head1', 'head2'],
		}
		self.pred_symtr=pred_symtr
		self.task_type = task_type
		self.per_obj = per_obj
		self.symmetric_ee = ['hammer_grip_head1', 'screwdriver_head1', 'wrench_head1', 'wrench_head2']
		self.mode = mode
		assert self.mode in ['train', 'test', 'val', 'novel'], "mode should be either 'train' or 'test'"
		self.data_dir = data_dir
		self.n_pts = n_pts
		if self.mode == 'train':
			self.meta_list = np.loadtxt(os.path.join(data_dir, f'train_examples_ee_visible.txt'), delimiter=',', dtype=str)
		elif self.mode == 'test' or self.mode == 'val':
			self.meta_list = np.loadtxt(os.path.join(data_dir, f'val_examples_ee_visible.txt'), delimiter=',', dtype=str)
		elif self.mode == 'novel':
			self.meta_list = np.loadtxt(os.path.join(data_dir, f'novel_examples_ee_visible.txt'), delimiter=',', dtype=str)
		
		assert self.meta_list is not None, "data_list is None"
		if self.per_obj is not None:
			self.meta_list = [item for item in self.meta_list if self.per_obj in item[0]]
		self.length = len(self.meta_list)
	
	def __getitem__(self, index):
		'''
			batch sample keys:
				batch size is 192
				'pcl_in': points cloud data, (N,1024,3)
				'cat_id': category ID (N,)
				'R': Rotation Matrix (N,3,3)
				'T': Translation vector, Tx, Ty, Tz (N,3)
		'''

		data_dict = {}
		meta_name, ee_name = self.meta_list[index]
		meta_utils = MetaUtils(data_root=self.data_dir, meta_name=meta_name)
		cat_name, obj_name, inst_id = meta_utils.get_cat_obj_id()
		ee_pose = meta_utils.get_ee_poses(ee_name=ee_name)
		obj_scale = meta_utils.get_obj_scale()

		if self.task_type == 'obj_pose':
			obj_pose = meta_utils.get_obj_pose()
			obj_points, obj_colors = meta_utils.get_obj_point_cloud()
			pcd_path = meta_utils.obj_pcd_path
			pcl_in = obj_points
			gt_pose = obj_pose
			colors_in = obj_colors*255
			
		elif self.task_type == 'ee_pose':
			ee_points, ee_colors = meta_utils.get_ee_pcd(ee_name)
			gt_pose = ee_pose
			pcl_in = ee_points
			pcd_path = meta_utils.get_ee_pcd_path(ee_name)
			colors_in = ee_colors*255
		else:
			raise NotImplementedError
		
		if pcl_in.shape[0] != 1024:
			pcl_in, idx = sample_data(pcl_in, self.n_pts)  # (1024,3)
			colors_in = colors_in[idx]
		affrdn_ee_name = f"{cat_name}_{ee_name}"
		if self.pred_symtr:
			if affrdn_ee_name in self.symmetric_ee:
				data_dict['symtr'] = torch.as_tensor(np.array([0,0,1])).contiguous()  # denote the z axis is the symmetry axis
			else:
				data_dict['symtr'] = torch.as_tensor(np.array([0,0,0])).contiguous()  # denote there are no symmetric axis
		data_dict['pcl_in'] = torch.as_tensor(pcl_in.astype(np.float32)).contiguous()
		data_dict['colors_in'] = torch.as_tensor(colors_in.astype(np.uint8)).contiguous()
		data_dict['gt_RTs'] = torch.as_tensor(gt_pose, dtype=torch.float32).contiguous()  # (4,4) target pose
		data_dict['ee_RTs'] = torch.as_tensor(ee_pose, dtype=torch.float32).contiguous()  # (4,4) target pose
		data_dict['cat_id'] = torch.as_tensor(self.cat_name_to_id[cat_name], dtype=torch.int8).contiguous()
		data_dict['affrdn_ee_name'] = affrdn_ee_name
		data_dict['obj_scale'] = obj_scale
		data_dict['obj_name'] = obj_name
		data_dict['pcd_path'] = pcd_path
		data_dict['meta_name'] = meta_name
		data_dict['cat_name'] = cat_name
		data_dict['inst_id'] = inst_id
		data_dict['gt_rotation'] = torch.as_tensor(gt_pose[:3, :3], dtype=torch.float32).contiguous()  # (3,3) rotation matrix
		data_dict['gt_translation'] = torch.as_tensor(gt_pose[:3, 3], dtype=torch.float32).contiguous()  # (3,) translation vector, xyz
		# print the data shape for each batch
		assert data_dict['pcl_in'].shape == torch.Size([1024, 3]), f"pcl_in shape is {data_dict['pcl_in'].shape}"
		assert data_dict['gt_RTs'].shape == torch.Size([4, 4]), f"gt_RTs shape is {data_dict['gt_RTs'].shape}"
		assert data_dict['ee_RTs'].shape == torch.Size([4, 4]), f"ee_RTs shape is {data_dict['ee_RTs'].shape}"
		assert data_dict['gt_rotation'].shape == torch.Size(
			[3, 3]), f"gt_rotation shape is {data_dict['gt_rotation'].shape}"
		assert data_dict['gt_translation'].shape == torch.Size(
			[3]), f"gt_translation shape is {data_dict['gt_translation'].shape}"
		return data_dict
	
	# the length of the dataset
	def __len__(self):
		return self.length


# wrap the dataset into the pytorch dataloader
def get_data_loaders(
		batch_size,
		seed,
		percentage_data=1.0,
		data_path=None,
		mode='train',  # train, test, val, novel
		n_pts=1024,
		per_obj=None,
		num_workers=32,
		task_type='ee_pose',
		pred_symtr=False,
):
	# set up the random seed for reproducibility
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	
	# load the dataset
	dataset = TooleeDataset(
		mode=mode,
		data_dir=data_path,
		n_pts=n_pts,
		per_obj=per_obj,
		task_type=task_type,
		pred_symtr=pred_symtr,
	)
	num_workers = num_workers
	if mode == 'train':
		shuffle = True
		
	else:
		shuffle = False
		
	# wrap the dataset into the pytorch dataloader for training
	if mode == 'train':
		idx = list(range(len(dataset)))
		random.shuffle(idx)
		size = int(percentage_data * len(idx))
		idx = idx[:size]
		data_sampler = torch.utils.data.sampler.SubsetRandomSampler(
			idx)  # sampler for not put it back, eahc time choice different idx
		# put the dataset into the pytorch dataloader
		dataloader = torch.utils.data.DataLoader(
			dataset,
			batch_size=batch_size,
			sampler=data_sampler,
			num_workers=num_workers,
			persistent_workers=True,
			drop_last=False,
			pin_memory=True,
		)
	# wrap the dataset into the pytorch dataloader for testing or validation
	else:
		# sample
		size = int(percentage_data * len(dataset))
		# return n split dataset, but only take the first one.
		dataset, _ = torch.utils.data.random_split(dataset, (size, len(dataset) - size))
		# train_dataloader = torch.utils.data.DataLoader(
		dataloader = torch.utils.data.DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=shuffle,
			num_workers=num_workers,
			persistent_workers=True,
			drop_last=False,
			pin_memory=True,
		)
	return dataloader


# read the config file and call the dataloader function respectively
def get_data_loaders_from_cfg(cfg, data_type=['train', 'val', 'test', 'novel']):
	data_loaders = {}
	pred_symtr = True if cfg.regression_head == 'Rx_Ry_and_T_and_Symtr' else False
	
	if 'train' in data_type:
		train_loader = get_data_loaders(
			batch_size=cfg.batch_size,
			seed=cfg.seed,
			percentage_data=cfg.percentage_data_for_train,  # default 1.0
			data_path=cfg.data_path,
			mode='train',
			n_pts=cfg.num_points,  # default 1024, sample points for points cloud
			per_obj=cfg.per_obj,  # default '', if specified, only train one object from ['Hammer_Grip']
			num_workers=cfg.num_workers,
			task_type=cfg.task_type,
			pred_symtr=pred_symtr,
		)
		data_loaders['train_loader'] = train_loader
	
	if 'test' in data_type:
		test_loader = get_data_loaders(
			batch_size=cfg.eval_batch_size,
			seed=cfg.seed,
			percentage_data=cfg.percentage_data_for_test,
			data_path=cfg.data_path,
			mode='test',
			n_pts=cfg.num_points,
			per_obj=cfg.per_obj,
			num_workers=cfg.num_workers,
			task_type=cfg.task_type,
			pred_symtr=pred_symtr,
		)
		data_loaders['test_loader'] = test_loader
	
	if 'val' in data_type:
		val_loader = get_data_loaders(
			batch_size=cfg.eval_batch_size,
			seed=cfg.seed,
			percentage_data=cfg.percentage_data_for_val,
			data_path=cfg.data_path,
			mode='val',
			n_pts=cfg.num_points,
			per_obj=cfg.per_obj,
			num_workers=cfg.num_workers,
			task_type=cfg.task_type,
			pred_symtr=pred_symtr,
		)
		data_loaders['val_loader'] = val_loader
	
	if 'novel' in data_type:
		novel_loader = get_data_loaders(
			batch_size=cfg.eval_batch_size,
			seed=cfg.seed,
			percentage_data=cfg.percentage_data_for_test,
			data_path=cfg.data_path,
			mode='novel',
			n_pts=cfg.num_points,
			per_obj=cfg.per_obj,
			num_workers=cfg.num_workers,
			task_type=cfg.task_type,
			pred_symtr=pred_symtr,
		)
		data_loaders['novel_loader'] = novel_loader
	
	return data_loaders

if __name__ == '__main__':
	cfg = get_config()
	cfg.data_path = '/dataSSD/yunlong/dataspace/DatasetToolEE'
	cfg.asset_path = '/dataSSD/yunlong/dataspace/Dataset3DModel'
	cfg.task_type = 'ee_pose'
	cfg.pose_mode = 'rot_matrix'
	cfg.batch_size = 600
	cfg.eval_batch_size = 200
	cfg.num_workers = 64
	is_pred_symtr = True if cfg.regression_head == 'Rx_Ry_and_T_and_Symtr' else False
	
	# cfg.device = "cuda"
	data_loaders = get_data_loaders_from_cfg(cfg, data_type=['train', 'val', 'test', 'novel'])
	train_loader = data_loaders['train_loader']
	test_loader = data_loaders['test_loader']
	val_loader = data_loaders['val_loader']
	novel_loader = data_loaders['novel_loader']
	for key, value in cfg.__dict__.items():
		print(f'{key}: {value}')
		
	for index, batch_sample in enumerate(tqdm(train_loader, desc='train_loader')):
		batch_sample = process_batch(
			batch_sample=batch_sample,
			device=cfg.device,
			is_pred_symtr=is_pred_symtr,
		)

	for index, batch_sample in enumerate(tqdm(test_loader, desc='test_loader')):
		batch_sample = process_batch(
			batch_sample=batch_sample,
			device=cfg.device,
			is_pred_symtr=is_pred_symtr,
		)

	for index, batch_sample in enumerate(tqdm(val_loader, desc='val_loader')):
		batch_sample = process_batch(
			batch_sample=batch_sample,
			device=cfg.device,
			is_pred_symtr=is_pred_symtr
		)

	for index, batch_sample in enumerate(tqdm(novel_loader, desc='novel_loader')):
		batch_sample = process_batch(
			batch_sample=batch_sample,
			device=cfg.device,
			is_pred_symtr=is_pred_symtr
		)
		
