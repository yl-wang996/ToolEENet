import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
host_name = "tams110"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["ROS_MASTER_URI"] = f"http://{host_name}:11311"

import numpy as np
from utils.misc import get_rot_matrix
from networks.posenet_agent import PoseNet
from configs.config import get_config
from utils.transform_utils import TfUtils
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from tool_ee.srv import PosePred, PosePredResponse
from geometry_msgs.msg import Pose
import torch
import copy
import tf

def pc2_msg_to_points_array(pc2_msg: PointCloud2):
	return np.array(list(point_cloud2.read_points(pc2_msg, field_names=("x", "y", "z"))))

def process_batch(points, device='cuda'):
	
	"""
	process the batch data
	:param batch_sample: a batch of data from dataloader
	:param device: device to store the data
	:return: processed batch data
	"""
	points = points.to(device)
	processed_sample = {}
	processed_sample['pts'] = points  # [bs, 1024, 3]
	
	""" zero center """
	# move the pts to the zero center
	num_pts = processed_sample['pts'].shape[1]
	zero_mean = torch.mean(processed_sample['pts'][:, :, :3], dim=1)
	processed_sample['zero_mean_pts'] = copy.deepcopy(processed_sample['pts'])
	processed_sample['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
	processed_sample['pts_center'] = zero_mean
	return processed_sample

def get_infer_ee_pose_cfg():
	''' load config '''
	cfg = get_config(show=False)
	epoch = 2000
	cfg.score_model_dir = f'ScoreNet/ckpt_epoch{epoch}.pth'
	cfg.task_type = 'ee_pose'
	cfg.use_symtr_prior = False
	cfg.regression_head = 'Rx_Ry_and_T_and_Symtr'  # Rx_Ry_and_T, Rx_Ry_and_T_and_Symtr
	cfg.pose_mode = 'rot_matrix_symtr'  # rot_matrix_symtr, rot_matrix
	cfg.log_folder = f"/dataSSD/yunlong/dataspace/training_logs_ee_pose_symtr"
	cfg.eval_repeat_num = 20
	cfg.eval_set = "test"  # test, novel
	cfg.data_path = "/dataSSD/yunlong/dataspace/DatasetToolEE"
	cfg.sampler_mode = ["ode"]
	cfg.max_eval_num = 1000000
	cfg.percentage_data_for_test = 1.0
	cfg.batch_size = 200
	cfg.seed = 0
	cfg.T0 = 0.55
	cfg.num_gpu = 1
	for k, v in cfg.__dict__.items():
		print(f'{k}: {v}')
	return cfg

def get_infer_obj_pose_cfg():
	''' load config '''
	cfg = get_config(show=False)
	epoch = 1999
	cfg.score_model_dir = f'ScoreNet/ckpt_epoch{epoch}.pth'
	cfg.task_type = 'obj_pose'
	cfg.use_symtr_prior = False
	cfg.regression_head = 'Rx_Ry_and_T'  # Rx_Ry_and_T, Rx_Ry_and_T_and_Symtr
	cfg.pose_mode = 'rot_matrix'  # rot_matrix_symtr, rot_matrix
	cfg.log_folder = f"/dataSSD/yunlong/dataspace/training_logs_obj_pose"
	cfg.eval_repeat_num = 20
	cfg.eval_set = "test"  # test, novel
	cfg.data_path = "/dataSSD/yunlong/dataspace/DatasetToolEE"
	cfg.sampler_mode = ["ode"]
	cfg.max_eval_num = 1000000
	cfg.percentage_data_for_test = 1.0
	cfg.batch_size = 200
	cfg.seed = 0
	cfg.T0 = 0.55
	cfg.num_gpu = 1
	for k, v in cfg.__dict__.items():
		print(f'{k}: {v}')
	return cfg

class ToolEEPredictor(object):
	def __init__(self, cfg):
		self.cfg = cfg
		self.score_agent = None
	
	def init_model(self, ckpt_path=None):
		self.cfg.posenet_mode = 'score'
		if ckpt_path is None:
			ckpt_path = os.path.join(self.cfg.log_folder, f'results/ckpts/{self.cfg.score_model_dir}')
			assert os.path.exists(ckpt_path), f"ScoreNet checkpoint {ckpt_path} does not exist!"
		self.score_agent = PoseNet(self.cfg)
		self.score_agent.load_ckpt(model_dir=ckpt_path, model_path=True, load_model_only=True)
	
	def predict(self, points):
		''' predict poses '''
		# data: [bs, 1024, 3]
		if isinstance(points, np.ndarray):
			points = torch.tensor(points, dtype=torch.float32)
		assert self.score_agent is not None, "ScoreNet model is not loaded!"
		porcessed_data = process_batch(
			points=points,
			device=self.cfg.device,
		)
		# [bs, repeat_num, 4, 4], [bs, repeat_num, 9], [bs,3]
		pred_RTs, _, pred_symtrs = self.pred_pose_batch(porcessed_data)  # poses (6+3)vector,  RTs(4,4) matrix
		avg_pred_RT = TfUtils.get_avg_sRT(pred_RTs)
		if pred_symtrs is not None:
			pred_symtrs = np.average(pred_symtrs, axis=1)
		else:
			pred_symtrs = np.repeat(np.array([0, 0, 0])[np.newaxis, ...], points.shape[0], axis=0)
		# pred_symtrs[pred_symtrs < symtr_threshold] = 0
		# pred_symtrs[pred_symtrs >= symtr_threshold] = 1
		rospy.logdebug(f"Predicted RT: {avg_pred_RT}, pred_symtrs: {pred_symtrs}")
		return avg_pred_RT, pred_symtrs
	
	def pred_pose_batch(self, batch_sample):
		"""
		Predict poses for a batch of data
		Args:
			batch_sample: a batch of data
		Returns:
			RTs_all: [bs, repeat_num, 4, 4]
			pred_pose: [bs, repeat_num, 6+3+3] # 6d rotation, 3d translation, 3d symtr
			pred_symtrs: [bs, repeat_num, 3]
		"""
		pred_symtrs = None
		assert self.score_agent is not None, "ScoreNet model is not loaded!"
		pred_pose, _, _, _ = self.score_agent.pred_func(
			data=batch_sample,
			repeat_num=self.cfg.eval_repeat_num,
			T0=self.cfg.T0,
		)
		
		if pred_pose.shape[2] == 12:
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

class PredictNode(object):
	def __init__(self,cfg):
		self.cfg = cfg
		self.predictor = ToolEEPredictor(self.cfg)
	
	def start_service(self,srv_name='pose_pred'):
		''' spin up the node '''
		rospy.Service(srv_name, PosePred, self.prediction_service)
	
	def prediction_service(self, pc2_msg):
		''' process pointcloud '''
		points = pc2_msg_to_points_array(pc2_msg.pc2)
		if len(points.shape) < 3:
			points = points[np.newaxis, ...]
		pred_RT, pred_symtrs = self.predictor.predict(points)
		pose_msg = self.RT_to_Pose_msg(pred_RT[0])
		return PosePredResponse(pred_pose=pose_msg, pred_symtrs=pred_symtrs[0] if pred_symtrs is not None else [])
	
	def RT_to_Pose_msg(self,RT):
		''' convert RT to geometry_msgs/Pose msg'''
		trans, quat = TfUtils.decompose_tf_M(RT)
		pose_msg = Pose()
		pose_msg.position.x = trans[0]
		pose_msg.position.y = trans[1]
		pose_msg.position.z = trans[2]
		pose_msg.orientation.x = quat[0]
		pose_msg.orientation.y = quat[1]
		pose_msg.orientation.z = quat[2]
		pose_msg.orientation.w = quat[3]
		return pose_msg
	
	def Pose_msg_to_RT(self,pose_msg):
		''' convert geometry_msgs/Pose msg to RT'''
		RT = TfUtils.compose_tf_M(np.asarray(pose_msg.position),np.asarray(pose_msg.orientation))
		return RT
	
	def warm_up(self):
		''' warm up the model '''
		rospy.loginfo("Warming up the model...")
		self.predictor.init_model()
		fake_points = np.zeros((1, 1024, 3))
		fake_points = torch.tensor(fake_points, dtype=torch.float32)
		self.predictor.predict(fake_points)
	
if __name__ == '__main__':
	rospy.init_node('pose_prediction_node',log_level=rospy.DEBUG)
	# infer_cfg = get_infer_ee_pose_cfg()
	infer_cfg = get_infer_obj_pose_cfg()
	rospy.loginfo(f"Starting pose prediction service with task_type:{infer_cfg.task_type}...")
	pred_node = PredictNode(infer_cfg)
	pred_node.warm_up()
	pred_node.start_service('pose_pred')
	try:
		rospy.wait_for_service('pose_pred', timeout=10)
		rospy.loginfo('pose_pred service is available')
	except rospy.ROSException:
		rospy.logerr('pose_pred service not available')
		rospy.signal_shutdown('pose_pred service not available')
	rospy.spin()
