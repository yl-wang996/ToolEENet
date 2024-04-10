import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

import _pickle as cPickle
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.misc import exists_or_mkdir, get_rot_matrix
from utils.metrics import compute_mAP, plot_mAP

from networks.posenet_agent import PoseNet
from configs.config import get_config
from datasets.dataset_toolee import get_data_loaders_from_cfg, process_batch
from utils.transform_utils import TfUtils


''' load config '''
cfg = get_config()
epoch = 2000
cfg.score_model_dir = f'ScoreNet/ckpt_epoch{epoch}.pth'
cfg.task_type = 'ee_pose'
cfg.use_symtr_prior = False
# cfg.regression_head = 'Rx_Ry_and_T'  # Rx_Ry_and_T, Rx_Ry_and_T_and_Symtr
cfg.regression_head = 'Rx_Ry_and_T_and_Symtr'  # Rx_Ry_and_T, Rx_Ry_and_T_and_Symtr
# cfg.pose_mode = 'rot_matrix'  # rot_matrix_symtr, rot_matrix
cfg.pose_mode = 'rot_matrix_symtr'  # rot_matrix_symtr, rot_matrix
# cfg.log_folder = f"/dataSSD/yunlong/dataspace/training_logs_ee_pose"
cfg.log_folder = f"/dataSSD/yunlong/dataspace/training_logs_ee_pose_symtr"
cfg.eval_repeat_num = 20
cfg.eval_set = "test"  # test, novel
overwrite = False

cfg.data_path = "/dataSSD/yunlong/dataspace/DatasetToolEE"
cfg.sampler_mode = ["ode"]
cfg.max_eval_num = 1000000
cfg.percentage_data_for_test = 1.0
cfg.batch_size = 200
cfg.seed = 0
cfg.T0 = 0.55
cfg.num_gpu = 1
is_pred_symtr = True if cfg.regression_head == 'Rx_Ry_and_T_and_Symtr' else False

''' create checkpoint list '''
scorenet_ckpt_path = os.path.join(cfg.log_folder, f'results/ckpts/{cfg.score_model_dir}')
assert os.path.exists(scorenet_ckpt_path), f"ScoreNet checkpoint {scorenet_ckpt_path} does not exist!"
''' create result dir '''

inference_res_root_dir = os.path.join(cfg.log_folder, f'{cfg.eval_set}_results')
exists_or_mkdir(inference_res_root_dir)
inference_res_dir = os.path.join(inference_res_root_dir, f'infer_repeat_{cfg.eval_repeat_num}')
exists_or_mkdir(inference_res_dir)

def record_results_and_draw_curves(save_path, avg_percision_dict, avg_acc_dict, symtr_acc_dict, degree_thres_list, shift_thres_list):
	# draw curves
	plot_mAP(
		avg_percision_dict=avg_percision_dict,
		out_dir=save_path,
		degree_thres_list=degree_thres_list,
		shift_thres_list=shift_thres_list,
		out_name=f"avg_mAP.png"
	)
	# record results
	degree_05_idx = degree_thres_list.index(5)
	degree_10_idx = degree_thres_list.index(10)
	shift_01_idx = shift_thres_list.index(1)
	shift_02_idx = shift_thres_list.index(2)
	shift_05_idx = shift_thres_list.index(5)
	cls_names = list(avg_percision_dict.keys())
	for cls_name in cls_names:
		pose_ap = np.asarray(avg_percision_dict[cls_name])
		pose_acc = np.asarray(avg_acc_dict[cls_name])
		
		messages = []
		messages.append(f"cls_name: {cls_name}")
		
		
		messages.append('mAP:')
		messages.append('5 degree, 1cm: {:.1f}'.format(pose_ap[degree_05_idx, shift_01_idx] * 100))
		messages.append('5 degree, 2cm: {:.1f}'.format(pose_ap[degree_05_idx, shift_02_idx] * 100))
		messages.append('5 degree, 5cm: {:.1f}'.format(pose_ap[degree_05_idx, shift_05_idx] * 100))
		
		messages.append('10 degree, 1cm: {:.1f}'.format(pose_ap[degree_10_idx, shift_01_idx] * 100))
		messages.append('10 degree, 2cm: {:.1f}'.format(pose_ap[degree_10_idx, shift_02_idx] * 100))
		messages.append('10 degree, 5cm: {:.1f}'.format(pose_ap[degree_10_idx, shift_05_idx] * 100))
		
		
		messages.append('Acc:')
		messages.append('5 degree, 1cm: {:.1f}'.format(pose_acc[degree_05_idx, shift_01_idx] * 100))
		messages.append('5 degree, 2cm: {:.1f}'.format(pose_acc[degree_05_idx, shift_02_idx] * 100))
		messages.append('5 degree, 5cm: {:.1f}'.format(pose_acc[degree_05_idx, shift_05_idx] * 100))
		
		messages.append('10 degree, 1cm: {:.1f}'.format(pose_acc[degree_10_idx, shift_01_idx] * 100))
		messages.append('10 degree, 2cm: {:.1f}'.format(pose_acc[degree_10_idx, shift_02_idx] * 100))
		messages.append('10 degree, 5cm: {:.1f}'.format(pose_acc[degree_10_idx, shift_05_idx] * 100))
		
		if symtr_acc_dict is not None:
			messages.append('Symtr Acc:')
			symtr_acc = symtr_acc_dict[cls_name]
			messages.append(f"{cls_name} acc: {symtr_acc*100:.1f}")

		with open(os.path.join(save_path, f"eval_logs_{cls_name}.txt"), 'w') as f:
			print('-'*30)
			for msg in messages:
				print(msg)
				f.write(msg + '\n')

def unpack_data(path):
	detect_result_path = path
	with open(detect_result_path, 'rb') as f:
		detect_result = cPickle.load(f)
	
	categorized_test_data = {}
	for cat_name in cfg.synset_names:
		categorized_test_data[cat_name] = {'img_path': [],
										   'pts': [],
										   'rgb': [],
										   'cat_id': [],
										   'inst': [], }
	
	print('Extracting data...')
	for key in tqdm(detect_result.keys()):
		instance_num = detect_result[key]['result']['pred_RTs'].shape[0]
		# detect_result[key]['result']['choosed_pred_RTs'] = copy.deepcopy(detect_result[key]['result']['pred_RTs'])
		detect_result[key]['result']['multi_hypothesis_pred_RTs'] = \
			np.identity(4, dtype=float)[np.newaxis, np.newaxis, ...].repeat(instance_num, axis=0).repeat(
				cfg.eval_repeat_num, axis=1)
		detect_result[key]['result']['energy'] = \
			np.zeros(2, dtype=float)[np.newaxis, np.newaxis, ...].repeat(instance_num, axis=0).repeat(
				cfg.eval_repeat_num, axis=1)
		
		# result = detect_result[key]['result']
		valid_pts = detect_result[key]['valid_pts']
		valid_rgb = detect_result[key]['valid_rgb']
		cat_id = detect_result[key]['cat_id']
		valid_inst = detect_result[key]['valid_inst']
		
		if len(valid_inst):
			for i in range(len(valid_inst)):
				cat_name = cfg.synset_names[cat_id[i]]
				categorized_test_data[cat_name]['img_path'].append(key)
				categorized_test_data[cat_name]['pts'].append(valid_pts[i])
				categorized_test_data[cat_name]['cat_id'].append(cat_id[i])
				categorized_test_data[cat_name]['inst'].append(valid_inst[i])
				if not valid_rgb is None:
					categorized_test_data[cat_name]['rgb'].append(valid_rgb[i])
				else:
					categorized_test_data[cat_name]['rgb'] = None
	return detect_result, categorized_test_data

def pred_pose_batch(score_agent: PoseNet, batch_sample):
	''' Predict poses '''
	pred_symtrs = None
	pred_pose, _, _, _ = score_agent.pred_func(
		data=batch_sample,
		repeat_num=cfg.eval_repeat_num,
		T0=cfg.T0,
	)
	
	if pred_pose.shape[2]==12:
		pred_symtrs = pred_pose[:, :, -3:]
		pred_symtrs = pred_symtrs.cpu().numpy()
		
	''' Transfer predicted poses (6+3)vector to RTs(4,4) matrix '''
	RTs_all = np.ones((pred_pose.shape[0], pred_pose.shape[1], 4, 4))  # [bs, repeat_num, 4, 4]
	for i in range(pred_pose.shape[1]):
		R = get_rot_matrix(pred_pose[:, i, :6])
		T = pred_pose[:, i, 6:9]
		RTs = np.identity(4, dtype=float)[np.newaxis, ...].repeat(R.shape[0], 0)
		RTs[:, :3, :3] = R.cpu().numpy()
		RTs[:, :3, 3] = T.cpu().numpy()
		RTs_all[:, i, :, :] = RTs
	return RTs_all, pred_pose, pred_symtrs

def apply_pose_prior(pose_prior, pred_RT, scale):
	# rescale the ee pose prior according to the domain randomization
	pose_prior = np.asarray(pose_prior)
	scale = np.asarray(scale)
	prior_trans, prior_quat = TfUtils.decompose_tf_M(pose_prior)
	prior_trans = prior_trans*scale
	pose_prior = TfUtils.compose_tf_M(trans=prior_trans, quat=prior_quat)
	refined_pred_RT = np.asarray(pose_prior) @ np.asarray(pred_RT)
	return refined_pred_RT

def infer_pose(dataset='test', result_path=None):
	''' Create evaluation agent '''
	cfg.posenet_mode = 'score'
	score_agent = PoseNet(cfg)
	assert dataset in ['test', 'novel'], 'dataset should be one of the [test, novel]'
	pose_priors = None
	
	''' Load model '''
	score_agent.load_ckpt(model_dir=scorenet_ckpt_path, model_path=True, load_model_only=True)
	pred_results = []
	data_loaders = get_data_loaders_from_cfg(cfg, data_type=['test', 'novel'])
	test_loader = data_loaders['test_loader']
	novel_loader = data_loaders['novel_loader']
	
	data_loader = test_loader if dataset == 'test' else novel_loader
	torch.cuda.empty_cache()
	for _, batch_sample in enumerate(tqdm(data_loader, desc=f'{dataset}_loader')):
		batch_size = batch_sample['pcl_in'].shape[0]
		porcessed_batch_sample = process_batch(
			batch_sample=batch_sample,
			device=cfg.device,
			is_pred_symtr=is_pred_symtr
		)
		# [bs, repeat_num, 4, 4], [bs, repeat_num, 9]
		pred_RTs, pred_poses, pred_symtrs = pred_pose_batch(score_agent, porcessed_batch_sample)  # poses (6+3)vector,  RTs(4,4) matrix
		# TODO. filter out the outlier here
		avg_pred_RT = TfUtils.get_avg_sRT(pred_RTs)
		
		pred_obj_RT = None
		pred_ee_RT = None
		gt_ee_RT = None
		gt_obj_RT = None
		for idx in range(batch_size):
			pred_RT = avg_pred_RT[idx]

			pred_ee_RT = pred_RT
			gt_ee_RT = batch_sample['ee_RTs'][idx]

			pred_results.append({
				'meta_name': batch_sample['meta_name'][idx],  # str
				'cat_name': batch_sample['cat_name'][idx],  # str
				'pred_ee_RT': pred_ee_RT,  # (4,4)
				'pred_obj_RT': pred_obj_RT,  # (4,4) is None, when cfg.task_type is ee_pose
				'gt_obj_RT': gt_obj_RT, # (4,4) is None, when cfg.task_type is ee_pose
				'gt_ee_RT': gt_ee_RT,  # (4,4)
				'affrdn_ee_name': batch_sample['affrdn_ee_name'][idx],
				'pred_symtr': pred_symtrs[idx] if pred_symtrs is not None else None,
			})
			
	with open(result_path, 'wb') as f:
		cPickle.dump(pred_results, f)
	f.close()

def evaluate(result_path, task_type='ee_pose', eval_taregt='ee_pose'):
	degree_thres_list = list(range(0, 15, 1))
	shift_thres_list = [i / 2 for i in range(21)]
	
	assert task_type == 'ee_pose'
	assert os.path.exists(result_path), f"Result path {result_path} does not exist!"
	with open(os.path.join(result_path), 'rb') as f:
		pred_results = cPickle.load(f)
	f.close()
	
	fw = open(f"{inference_res_dir}/eval_logs.txt", 'a')
	fw.write(f"score_model: {cfg.score_model_dir}" + '\n')
	fw.close()
	out_dir = os.path.join(inference_res_dir, f"task_{task_type}", f"target_{eval_taregt}")
	os.makedirs(out_dir, exist_ok=True)
	# get the ap and acc for each synset and the mean
	avg_percision_dict, avg_acc_dict, symtr_acc_dict = compute_mAP(
		pred_results=pred_results,
		out_dir=out_dir,
		degree_thresholds=degree_thres_list,
		shift_thresholds=shift_thres_list,
		pose_mode=cfg.pose_mode,
		use_symtr_prior=cfg.use_symtr_prior,
	)
	
	record_results_and_draw_curves(
		out_dir,
		avg_percision_dict,
		avg_acc_dict,
		symtr_acc_dict,
		degree_thres_list,
		shift_thres_list,
	)

def main():
	assert cfg.task_type != 'obj_pose'
	result_path = os.path.join(inference_res_dir, f'results.pkl')
	
	
	if os.path.exists(result_path) and not overwrite:
		print(f"Result path {result_path} already exists!")
	else:
		print('Predict pose ...')
		infer_pose(dataset=cfg.eval_set, result_path=result_path)
	
	print('Evaluating ...')
	evaluate(result_path=result_path, task_type=cfg.task_type, eval_taregt='ee_pose')
	
if __name__ == '__main__':
	main()