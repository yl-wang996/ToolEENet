import json
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import utils.transforms as transforms
import torch

from tqdm import tqdm

# from datasets.datasets_nocs import get_data_loaders_from_cfg, process_batch
# from datasets.datasets_genpose import get_data_loaders_from_cfg, process_batch
from datasets.dataset_toolee import get_data_loaders_from_cfg, process_batch
from networks.posenet_agent import PoseNet
from configs.config import get_config
from utils.misc import exists_or_mkdir
from utils.genpose_utils import merge_results
from utils.misc import average_quaternion_batch
from utils.metrics import get_errors, get_rot_matrix
from utils.so3_visualize import visualize_so3
from utils.visualize import create_grid_image


def prediction(cfg, dataloader, agent):
	pred_symtr = True if cfg.regression_head == 'Rx_Ry_and_T_and_Symtr' else False
	calc_confidence = False
	calc_energy = True
	if len(cfg.sampler_mode) != 1:
		raise NotImplementedError
	
	results = {}
	for index, test_batch in enumerate(tqdm(dataloader)):
		# inference a small batch samples
		if index > cfg.max_eval_num:
			break
		
		batch_sample = process_batch(
			batch_sample=test_batch,
			device=cfg.device,
			is_pred_symtr=pred_symtr
		)

		gt_rot_matrix = get_rot_matrix(batch_sample['gt_pose'][:, :6])  # get rot matrix from 6d rot pose
		gt_quat_wxyz = transforms.matrix_to_quaternion(gt_rot_matrix)
		
		pred_pose, average_pred_pose, choosed_pred_pose, energy = agent.pred_func(
			data=batch_sample,
			repeat_num=cfg.repeat_num,
			calc_confidence=calc_confidence,
			calc_energy=calc_energy)  # [bs, repeat_num, 7]
		
		result = {
			'pred_pose': pred_pose,
			'average_pred_pose': average_pred_pose,
			'choosed_pred_pose': choosed_pred_pose,
			'gt_pose': torch.cat((gt_quat_wxyz, batch_sample['gt_pose'][:, -3:]), dim=-1),
			'pts': batch_sample['pts'],
		}
		
		rot_error, trans_error,symtr_error = get_errors(
			result['average_pred_pose'],
			result['gt_pose'],
			class_ids=batch_sample['id'],
		)
		result['metrics'] = torch.cat(
			(torch.from_numpy(rot_error).reshape(-1, 1), torch.from_numpy(trans_error).reshape(-1, 1)), dim=-1)
		print('mean error: ', torch.mean(result['metrics'], dim=0))
		print('median error: ', torch.median(result['metrics'], dim=0).values)
		
		rot_error, trans_error,symtr_error = get_errors(
			result['choosed_pred_pose'],
			result['gt_pose'],
			class_ids=batch_sample['id'],
		)
		result['metrics'] = torch.cat(
			(torch.from_numpy(rot_error).reshape(-1, 1), torch.from_numpy(trans_error).reshape(-1, 1)), dim=-1)
		print('mean error: ', torch.mean(result['metrics'], dim=0))
		print('median error: ', torch.median(result['metrics'], dim=0).values)
		print('symtr error: ', symtr_error.mean())
		
		if index == 0:
			results = result
		else:
			for key in results.keys():
				if not results[key] == None:
					results[key] = torch.cat((results[key], result[key]), dim=0)
	
	''' results visualization '''
	for i in range(results['pred_pose'].shape[0]):
		gt_rot = results['gt_pose'][i][:6].unsqueeze(0)
		
		pred_rot = results['pred_pose'][i][:, :6]
		choosed_pred_rot = results['choosed_pred_pose'][i][:6].unsqueeze(0)
		average_pred_rot = results['average_pred_pose'][i][:6].unsqueeze(0)

		''' ToDo: render pointcloud '''
		grid_iamge, _ = create_grid_image(
			results['pts'][i].unsqueeze(0),
			results['average_pred_pose'][i].unsqueeze(0),
			results['gt_pose'][i].unsqueeze(0),
			None,
			pose_mode='quat_wxyz',
			inverse_pose=cfg.o2c_pose,
		)
		''' so3 distribution visualization '''
		visualize_so3(
			save_path='./so3_distribution.png',
			pred_rotations=get_rot_matrix(pred_rot).cpu().numpy(),
			pred_rotation=get_rot_matrix(average_pred_rot).cpu().numpy(),
			gt_rotation=get_rot_matrix(gt_rot).cpu().numpy(),
			image=grid_iamge,
			# probabilities=confidence
		)
		grid_iamge, _ = create_grid_image(
			results['pts'][i].unsqueeze(0),
			results['choosed_pred_pose'][i].unsqueeze(0),
			results['gt_pose'][i].unsqueeze(0),
			None,
			pose_mode='quat_wxyz',
			inverse_pose=cfg.o2c_pose,
		)
		visualize_so3(
			save_path='./so3_distribution.png',
			pred_rotations=get_rot_matrix(pred_rot).cpu().numpy(),
			pred_rotation=get_rot_matrix(choosed_pred_rot).cpu().numpy(),
			gt_rotation=get_rot_matrix(gt_rot).cpu().numpy(),
			image=grid_iamge,
			# probabilities=confidence
		)
	return results


def inference(cfg, dataloader, agent):
	if len(cfg.sampler_mode) != 1:
		raise NotImplementedError
	pred_symtr = True if cfg.regression_head == 'Rx_Ry_and_T_and_Symtr' else False
	repeat_num = cfg.repeat_num
	metrics = {}
	for i in range(repeat_num):
		epoch_rot_error = np.array([])
		epoch_trans_error = np.array([])
		epoch_results = {}
		pbar = tqdm(dataloader)
		pbar.set_description(f'NUM[{i + 1}/{repeat_num}]')
		for index, test_batch in enumerate(pbar):
			# inference a small batch samples
			if index > cfg.max_eval_num:
				break
			
			batch_sample = process_batch(
				batch_sample=test_batch,
				device=cfg.device,
				is_pred_symtr=pred_symtr
			)
			
			batch_metrics, sampler_mode, batch_results = agent.test_func(batch_sample, index)
			epoch_rot_error = np.concatenate([epoch_rot_error, batch_metrics['rot_error']['item']])
			epoch_trans_error = np.concatenate([epoch_trans_error, batch_metrics['trans_error']['item']])
			epoch_results = merge_results(epoch_results, batch_results)
			pbar.set_postfix({
				'MEAN_ROT_ERROR: ': batch_metrics['rot_error']['item'].mean(),
				'MEAN_TRANS_ERROR: ': batch_metrics['trans_error']['item'].mean()
			})
		
		pbar.set_postfix({
			'MEAN_ROT_ERROR: ': epoch_rot_error.mean(),
			'MEAN_TRANS_ERROR: ': epoch_trans_error.mean()
		})
		print("MEAM ROTATION ERROR: ", epoch_rot_error.mean())
		print("MEAN TRANSLATION ERROR: ", epoch_trans_error.mean())
		print("MEDIAN ROTATION ERROR: ", np.median(epoch_rot_error))
		print("MEDIAN TRANSLATION ERROR: ", np.median(epoch_trans_error))
		
		error = np.concatenate([epoch_rot_error[..., np.newaxis], epoch_trans_error[..., np.newaxis]], axis=1)
		metrics[f'index_{i}'] = error.tolist()
		
		if i == 0:
			results = epoch_results
			results['pred_pose'] = results['pred_pose'].unsqueeze(1)
		else:
			results['pred_pose'] = torch.cat([results['pred_pose'], epoch_results['pred_pose'].unsqueeze(1)], dim=1)
	
	''' Save metrics and results '''
	save_path = agent.model_dir.replace('ckpts', 'inference_results')
	save_path = os.path.join(
		save_path,
		cfg.test_source + '_' + sampler_mode + '_' + str(cfg.sampling_steps)
	)
	exists_or_mkdir(save_path)
	metrics_save_path = os.path.join(save_path, 'metrics.json')
	
	with open(metrics_save_path, 'w') as f:
		f.write(json.dumps(metrics, indent=1))
	f.close()
	
	''' Save results '''
	results_save_path = os.path.join(save_path, 'results.pkl')
	with open(results_save_path, 'wb') as f:
		pickle.dump(results, f)
	f.close()
	return results_save_path


def train_score(cfg, train_loader, val_loader, novel_loader, score_agent):
	""" Train score network or energy network without ranking
    Args:
        cfg (dict): config file
        train_loader (torch.utils.data.DataLoader): train dataloader
        val_loader (torch.utils.data.DataLoader): validation dataloader
        novel_loader (torch.utils.data.DataLoader): novel dataloader
        score_agent (torch.nn.Module): score network or energy network without ranking
    Returns:
    """
	is_pred_symtr = True if cfg.regression_head == 'Rx_Ry_and_T_and_Symtr' else False
	for epoch in range(score_agent.clock.epoch, cfg.n_epochs):
		''' train '''
		torch.cuda.empty_cache()
		# For each batch in the dataloader
		pbar = tqdm(train_loader)
		for i, batch_sample in enumerate(pbar):
			
			''' warm up'''
			if score_agent.clock.step < cfg.warmup:
				score_agent.update_learning_rate()
			
			''' load data '''
			batch_sample = process_batch(
				batch_sample=batch_sample,
				device=cfg.device,
				is_pred_symtr=is_pred_symtr
			)
			
			''' train score or energy without feedback'''
			losses = score_agent.train_func(data=batch_sample, gf_mode='score')
			
			pbar.set_description(
				f"EPOCH_{epoch}[{i}/{len(pbar)}][loss: {[value.item() for key, value in losses.items()]}]")
			score_agent.clock.tick()
		
		''' update learning rate and clock '''
		# if epoch >= 50 and epoch % 50 == 0:
		score_agent.update_learning_rate()
		score_agent.clock.tock()
		
		''' start eval '''
		if score_agent.clock.epoch % cfg.eval_freq == 0:
			rot_errors = []
			trans_errors = []
			data_loaders = [train_loader, val_loader, novel_loader]
			data_mode = 'val'
			for i, test_batch in enumerate(tqdm(val_loader, 'evaluating dataset')):
				test_batch = process_batch(
					batch_sample=test_batch,
					device=cfg.device,
					is_pred_symtr=is_pred_symtr,
				)
				metrics_bacth, sampler_mode_list = score_agent.eval_func(test_batch, data_mode)
				rot_errors.append(metrics_bacth[0]['rot_error']['mean'])
				trans_errors.append(metrics_bacth[0]['trans_error']['mean'])
				
			mean_rot_error = np.mean(np.asarray(rot_errors))
			mean_trans_error = np.mean(np.asarray(trans_errors))
			
			is_best = score_agent.is_best(rot_err=mean_rot_error, trans_err=mean_trans_error)
			print(f"Epoch: {epoch}, mean_rot_error: {mean_rot_error}, mean_trans_error: {mean_trans_error}")
			if is_best:
				score_agent.update_recorder(rot_err=mean_rot_error, trans_err=mean_trans_error)
				print(f"Best model at epoch {epoch}, mean_rot_error: {mean_rot_error}, mean_trans_error: {mean_trans_error}")
			''' save (ema) model '''
			score_agent.save_ckpt(is_best=is_best)

# parser the arguments, pass it to training agent for either scorenet
def main():
	# load config
	cfg = get_config()
	# cfg = setup_cfg_for_test(cfg)
	if cfg.pose_mode == 'rot_matrix_symtr':
		assert cfg.regression_head == 'Rx_Ry_and_T_and_Symtr', "The symmetric pose only support the regression head Rx_Ry_and_T_and_Symtr"
	''' Init data loaders'''
	data_loaders = get_data_loaders_from_cfg(cfg=cfg, data_type=['train', 'val', 'test', 'novel'])
	train_loader = data_loaders['train_loader']
	val_loader = data_loaders['val_loader']
	test_loader = data_loaders['test_loader']
	novel_loader = data_loaders['novel_loader']
	
	cfg.agent_type = 'score'
	cfg.posenet_mode = 'score'
	tr_agent = PoseNet(cfg)
	
	''' Load checkpoints '''
	load_model_only = False if cfg.use_pretrain else True
	if cfg.use_pretrain or cfg.eval or cfg.pred:
		tr_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=load_model_only)
	
	''' Start testing loop'''
	if cfg.eval:
		print("Start inference ...")
		inference(cfg, test_loader, tr_agent)
		print("Inference finished")
		exit()
	if cfg.pred:
		print("Start prediction ...")
		prediction(cfg, test_loader, tr_agent)
		print("Prediction finished")
		exit()
	
	''' Start training loop '''
	if cfg.agent_type == 'score':
		train_score(cfg, train_loader, val_loader, novel_loader, tr_agent)

def setup_cfg_for_test(cfg):
	cfg.data_path = "/dataSSD/yunlong/dataspace/DatasetToolEE"
	cfg.log_folder = "/dataSSD/yunlong/dataspace/training_logs_ee_pose_symtr_test"
	cfg.batch_size = 50
	cfg.eval_batch_size = 50
	cfg.log_dir = "ScoreNet"
	cfg.agent_type = "score"
	cfg.sampler_mode = "ode"
	cfg.sampling_steps = 500
	cfg.eval_freq = 1
	cfg.n_epochs = 2000
	cfg.percentage_data_for_train = 1.0
	cfg.percentage_data_for_test = 1.0
	cfg.seed = 0
	cfg.is_train = True
	cfg.task_type = "ee_pose"
	cfg.pose_mode = "rot_matrix_symtr"
	# cfg.pose_mode = "rot_matrix"
	cfg.regression_head = "Rx_Ry_and_T_and_Symtr"
	# cfg.regression_head = "Rx_Ry_and_T"
	return cfg

# TODO, consider how to incorporate the name embedding for conditional generation of the 6D pose
if __name__ == '__main__':
	main()
