import sys
import os

sys.path.append('..')

import torch
import numpy as np
import _pickle as cPickle

from utils.misc import get_rot_matrix, inverse_RT
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_RT_errors(pred_RT, gt_RT, pred_symtr=None, gt_symtr=None):
	"""
	Args:
		
	    pred_symtr: the symmetries of the object, if the object is symmetric, the rotation around the symmetry axis is not considered,
	        e.g. [0,0,1] denotes the object is symmetric around z-axis, then the rotation around z-axis is not considered.
		sRT_1: [4, 4]. homogeneous affine transformation
		sRT_2: [4, 4]. homogeneous affine transformation

	Returns:
		theta: angle difference of R in degree
		shift: l2 difference of T in centimeter
	"""
	# make sure the last row is [0, 0, 0, 1]
	if pred_RT is None or gt_RT is None:
		return -1
	try:
		assert np.array_equal(pred_RT[3, :], gt_RT[3, :])
		assert np.array_equal(pred_RT[3, :], np.array([0, 0, 0, 1]))
	except AssertionError:
		print(pred_RT[3, :], gt_RT[3, :])
		exit()
	
	R1 = pred_RT[:3, :3] / np.cbrt(np.linalg.det(pred_RT[:3, :3]))
	T1 = pred_RT[:3, 3]
	R2 = gt_RT[:3, :3] / np.cbrt(np.linalg.det(gt_RT[:3, :3]))
	T2 = gt_RT[:3, 3]
	
	symtr_acc = 0
	
	if pred_symtr is not None and gt_symtr is not None:
		symtr_acc = np.sum(pred_symtr == gt_symtr) / 3
		if 1 in pred_symtr:
			y = np.array(pred_symtr)
			y1 = R1 @ y
			y2 = R2 @ y
			cos_theta = y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
		else:
			R = R1 @ R2.transpose()
			cos_theta = (np.trace(R) - 1) / 2
	elif pred_symtr is None and gt_symtr is not None:
		if 1 in gt_symtr:
			y = np.array(gt_symtr)
			y1 = R1 @ y
			y2 = R2 @ y
			cos_theta = y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
		else:
			R = R1 @ R2.transpose()
			cos_theta = (np.trace(R) - 1) / 2
	elif pred_symtr is None and gt_symtr is None:
		R = R1 @ R2.transpose()
		cos_theta = (np.trace(R) - 1) / 2
	else:
		raise NotImplementedError
	
	theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
	shift = np.linalg.norm(T1 - T2) * 100
	result = np.array([theta, shift])
	return result, symtr_acc


def compute_RT_overlaps(class_ids, gt_RT, pred_RT, pred_symtr=None, gt_symtr=None):
	""" Finds overlaps between prediction and ground truth instances.

	Returns:
		overlaps:

	"""
	num = len(class_ids)
	overlaps = np.zeros((num, 2))
	symtr_acc = np.zeros(num)
	
	for i in range(num):
		if pred_symtr is not None:
			overlaps[i, :], symtr_acc[i] = compute_RT_errors(pred_RT[i], gt_RT[i], pred_symtr[i],gt_symtr[i])
		else:
			overlaps[i, :], symtr_acc[i] = compute_RT_errors(pred_RT[i], gt_RT[i])
	return overlaps, symtr_acc


# the function to get the rotation and translation error
def get_errors(pred_pose, gt_pose, class_ids):
	# 6D rotation representation by Zhou et al. [1] to rotation matrix.
	symtr_threshold = 0.5
	is_pred_symtr = pred_pose.shape[1] == 12
	pred_symtr = None
	gt_symtr = None
	symtr_error = None
	rot_1 = pred_pose[:, :6]
	rot_2 = gt_pose[:, :6]
	trans_1 = pred_pose[:, 6:9]
	trans_2 = gt_pose[:, 6:9]
	if is_pred_symtr:
		pred_symtr = pred_pose[:, 9:12]
		pred_symtr[pred_symtr<symtr_threshold]=0
		pred_symtr[pred_symtr>=symtr_threshold]=1
		gt_symtr = gt_pose[:, 9:12]
		gt_symtr[gt_symtr<symtr_threshold]=0
		gt_symtr[gt_symtr>=symtr_threshold]=1
		# symtr_error = torch.sum(pred_symtr == gt_symtr, axis=-1) / 3
	
	rot_matrix_1 = get_rot_matrix(rot_1)
	rot_matrix_2 = get_rot_matrix(rot_2)
	
	bs = pred_pose.shape[0]
	RT_1 = torch.eye(4).unsqueeze(0).repeat([bs, 1, 1])
	RT_2 = torch.eye(4).unsqueeze(0).repeat([bs, 1, 1])
	
	RT_1[:, :3, :3] = rot_matrix_1
	RT_1[:, :3, 3] = trans_1
	RT_2[:, :3, :3] = rot_matrix_2
	RT_2[:, :3, 3] = trans_2
	
	if pred_symtr is not None:
		error, symtr_acc = compute_RT_overlaps(
			class_ids=class_ids,
			gt_RT=RT_1.cpu().numpy(),
			pred_RT=RT_2.cpu().numpy(),
			pred_symtr=pred_symtr.cpu().numpy(),
			gt_symtr=gt_symtr.cpu().numpy()
		)
	else:
		error, symtr_acc = compute_RT_overlaps(
			class_ids=class_ids,
			gt_RT=RT_1.cpu().numpy(),
			pred_RT=RT_2.cpu().numpy()
		)
	rot_error = error[:, 0]  # in degree
	trans_error = error[:, 1]  # in centimeter
	
	return rot_error, trans_error, symtr_acc


def compute_RT_matches(overlap, degree_thres_list, shift_thres_list):
	assert overlap.shape[0] == 2
	
	num_degree_thres = len(degree_thres_list)
	num_shift_thres = len(shift_thres_list)
	
	_pred_match = np.zeros((num_degree_thres, num_shift_thres))
	_gt_match = np.zeros((num_degree_thres, num_shift_thres))
	d = degree_thres_list > overlap[0]
	s = shift_thres_list > overlap[1]
	_gt_match[d, :] += 1
	_pred_match[d, :] += 1
	_gt_match[:, s] += 1
	_pred_match[:, s] += 1
	pred_match = _pred_match == 2
	gt_match = _gt_match == 2
	# pred_match = pred_match.astype(np.int)
	# gt_match = gt_match.astype(np.int)
	return pred_match, gt_match


def compute_ap_acc(pred_matches: dict, gt_matches: dict):
	avg_percision_dict = {}
	avg_acc_dict = {}
	synset_names = list(pred_matches.keys())
	avg_percision_mean = []
	avg_acc_mean = []
	for synset_name in synset_names:
		pred_match = np.array(pred_matches[synset_name])
		gt_match = np.array(gt_matches[synset_name])
		avg_percision = np.sum(pred_match, axis=0) / pred_match.shape[0]
		avg_acc = np.sum(gt_match, axis=0) / gt_match.shape[0]
		avg_percision_dict[synset_name] = avg_percision.tolist()
		avg_acc_dict[synset_name] = avg_acc.tolist()
		avg_percision_mean.append(avg_percision)
		avg_acc_mean.append(avg_acc)
	avg_percision_dict['mean'] = np.mean(np.array(avg_percision_mean), axis=0).tolist()
	avg_acc_dict['mean'] = np.mean(np.array(avg_acc_mean), axis=0).tolist()
	return avg_percision_dict, avg_acc_dict


def compute_mAP(pred_results, out_dir, degree_thresholds=[180], shift_thresholds=[100], pose_mode='rot_matrix', use_symtr_prior=False):
	""" Compute mean Average Precision.
	Returns:
		pose_aps:
		pose_acc:
	"""
	assert pose_mode in ['rot_matrix_symtr', 'rot_matrix']
	is_pred_symtr = pose_mode == 'rot_matrix_symtr'

	synset_names = ['hammer_grip_head1', 'hammer_grip_grip', 'screwdriver_head1', 'wrench_head1', 'wrench_head2']
	symtr_names = ['hammer_grip_head1', 'screwdriver_head1', 'wrench_head1', 'wrench_head2']
	
	pred_matches = {}
	gt_matches = {}
	symtr_acc_dict = {}
	for k in synset_names:
		pred_matches[k] = []
		gt_matches[k] = []
		symtr_acc_dict[k] = []
	
	degree_thres_list = list(degree_thresholds) + [360]
	shift_thres_list = list(shift_thresholds) + [100]
	
	# loop over results to gather pred matches and gt matches for iou and pose metrics
	progress = 0
	
	for progress, result in enumerate(tqdm(pred_results)):
		
		synset_name = result['affrdn_ee_name']
		gt_RT = np.array(result['gt_ee_RT'])
		pred_RT = np.array(result['pred_ee_RT'])
		
		pred_symtr = None
		gt_symtr = None
		if is_pred_symtr:
			pred_symtr=np.array(result['pred_symtr'])
			if len(pred_symtr.shape)>1:
				pred_symtr = np.mean(pred_symtr, axis=0)
			pred_symtr[pred_symtr<0.5]=0
			pred_symtr[pred_symtr>=0.5]=1
		
		if use_symtr_prior or is_pred_symtr:
			if synset_name in symtr_names:
				gt_symtr = np.array([0, 0, 1])  # default symmetry axis is z-axis
			else:
				gt_symtr = np.array([0, 0, 0])
		
		if "wrench" in synset_name:
			gt_symtr = np.array([0, 1, 0])
			pred_symtr = np.array([0, 1, 0])
		RT_error, symtr_acc = compute_RT_errors(pred_RT, gt_RT, pred_symtr=pred_symtr, gt_symtr=gt_symtr)  # [theta, shift]
		pred_match, gt_match = compute_RT_matches(RT_error, degree_thres_list, shift_thres_list)
		
		pred_matches[synset_name].append(pred_match)
		gt_matches[synset_name].append(gt_match)
		if is_pred_symtr:
			symtr_acc_dict[synset_name].append(symtr_acc)
	
	avg_percision_dict, avg_acc_dict = compute_ap_acc(pred_matches=pred_matches,
	                                                  gt_matches=gt_matches)  # the ap and acc for each synset and the mean
	if is_pred_symtr:
		mean_symtr_acc = []
		for k,v in symtr_acc_dict.items():
			_mean = np.mean(v)
			symtr_acc_dict[k] = _mean
			mean_symtr_acc.append(_mean)
		symtr_acc_dict['mean'] = np.mean(mean_symtr_acc)
	else:
		symtr_acc_dict = None
	
	result_dict = {}
	result_dict['degree_thres_list'] = degree_thres_list
	result_dict['shift_thres_list'] = shift_thres_list
	result_dict['pose_aps'] = avg_percision_dict
	result_dict['pose_acc'] = avg_acc_dict
	result_dict['symtr_acc'] = symtr_acc_dict
	pkl_path = os.path.join(out_dir, f'mAP_Acc.pkl')
	with open(pkl_path, 'wb') as f:
		cPickle.dump(result_dict, f)
	return avg_percision_dict, avg_acc_dict, symtr_acc_dict


def plot_mAP(avg_percision_dict, out_dir, degree_thres_list, shift_thres_list, out_name='mAP.png'):
	labels = list(avg_percision_dict.keys())
	label_num = len(labels)
	colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:olive', 'tab:purple', 'tab:red', 'tab:gray']
	colors = colors[:label_num]
	styles = ['-'] * (label_num - 1) + ['--']
	
	fig, (ax_degree, ax_shift) = plt.subplots(1, 2, figsize=(8, 3.5))
	# rotation subplot
	ax_degree.set_title('Rotation', fontsize=10)
	ax_degree.set_ylim(0, 100)
	ax_degree.yaxis.set_ticks([0, 50, 60, 70, 80, 90, 100])
	ax_degree.set_xlabel('Degree')
	ax_degree.set_xlim(0, 10)
	ax_degree.xaxis.set_ticks([2, 5, 10])
	ax_degree.grid()
	for label, value in avg_percision_dict.items():
		value = np.array(value)
		i = labels.index(label)
		# y = 100 * np.average(value[:len(degree_thres_list), :], axis=1)
		y = 100 * value[:len(degree_thres_list), -1]
		ax_degree.plot(np.array(degree_thres_list), y,
		               color=colors[i], linestyle=styles[i], label=label)
	ax_degree.legend(loc='lower right', fontsize='small')
	
	# translation subplot
	ax_shift.set_title('Translation', fontsize=10)
	ax_shift.set_ylim(0, 100)
	ax_shift.set_xlim(0, 6)
	ax_shift.yaxis.set_ticks([0, 50, 60, 70, 80, 90, 100])
	ax_shift.set_xlabel('Centimeter')
	ax_shift.xaxis.set_ticks([1, 2, 4, 6])
	ax_shift.grid()
	for label, value in avg_percision_dict.items():
		value = np.array(value)
		i = labels.index(label)
		# y = 100 * np.average(value[:, :len(shift_thres_list)], axis=0)
		y = 100 * value[-1, :len(shift_thres_list)]
		ax_shift.plot(np.array(shift_thres_list), y,
		              color=colors[i], linestyle=styles[i], label=label)
	ax_shift.legend(loc='lower right', fontsize='small')
	
	plt.tight_layout()
	# plt.show()
	plt.savefig(os.path.join(out_dir, out_name), dpi=600)
	plt.close(fig)
	return