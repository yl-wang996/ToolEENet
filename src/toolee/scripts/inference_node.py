import os

host_name = "tams110"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["ROS_MASTER_URI"] = f"http://{host_name}:11311"
import sys
import _pickle as cPickle
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.misc import get_rot_matrix
from networks.posenet_agent import PoseNet
from configs.config import get_config
from datasets.dataset_toolee import get_data_loaders_from_cfg, process_batch
from utils.transform_utils import TfUtils
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import ros_numpy
from toolee_infer.srv import PosePred
import tf2_ros
from geometry_msgs.msg import Pose


def get_cfg():
	''' load config '''
	cfg = get_config()
	epoch = 2000
	cfg.score_model_dir = f'ScoreNet/ckpt_epoch{epoch}.pth'
	cfg.task_type = 'ee_pose'
	cfg.use_symtr_prior = False
	cfg.regression_head = 'Rx_Ry_and_T_and_Symtr'  # Rx_Ry_and_T, Rx_Ry_and_T_and_Symtr
	cfg.pose_mode = 'rot_matrix_symtr'  # rot_matrix_symtr, rot_matrix
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
	return cfg

class ToolEEPredictor(object):
	def __init__(self, cfg):
		self.cfg = cfg
		self.score_agent = None
	
	def init_model(self, ckpt_path=None):
		self.cfg.posenet_mode = 'score'
		if ckpt_path is None:
			ckpt_path = os.path.join(self.cfg.log_folder, f'results/ckpts/{cfg.score_model_dir}')
			assert os.path.exists(ckpt_path), f"ScoreNet checkpoint {ckpt_path} does not exist!"
		self.score_agent = PoseNet(cfg)
		self.score_agent.load_ckpt(model_dir=ckpt_path, model_path=True, load_model_only=True)
	
	def predict(self, data, is_pred_symtr=True):
		''' predict poses '''
		# data: [bs, 1024, 3]
		assert self.score_agent is not None, "ScoreNet model is not loaded!"
		porcessed_data = process_batch(
			batch_sample=data,
			device=cfg.device,
			is_pred_symtr=is_pred_symtr
		)
		# [bs, repeat_num, 4, 4], [bs, repeat_num, 9]
		pred_RTs, pred_poses, pred_symtrs = self.pred_pose_batch(porcessed_data)  # poses (6+3)vector,  RTs(4,4) matrix
		avg_pred_RT = TfUtils.get_avg_sRT(pred_RTs)
		return avg_pred_RT
	
	def pred_pose_batch(self, batch_sample):
		''' inference poses '''
		pred_symtrs = None
		assert self.score_agent is not None, "ScoreNet model is not loaded!"
		pred_pose, _, _, _ = self.score_agent.pred_func(
			data=batch_sample,
			repeat_num=cfg.eval_repeat_num,
			T0=cfg.T0,
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
	def __init__(self, cfg):
		self.cfg = get_cfg()
		self.predictor = ToolEEPredictor(cfg)
		# ros pointscloud2 listener
	
	def start_service(self):
		''' spin up the node '''
		rospy.Service('pose_pred', PosePred, self.prediction_service)
	
	def prediction_service(self, pc2_msg):
		''' process pointcloud '''
		points = self.pc2_to_array(pc2_msg)
		pred_RT = self.predictor.predict(points)
		pose_msg = self.RT_to_Pose_msg(pred_RT)
		return pose_msg
		
		
	def array_to_pc2(self,points):
		pc2_msg = ros_numpy.point_cloud2.array_to_pointcloud2(points, rospy.Time.now(), "camera_link")
		return pc2_msg
	
	def pc2_to_array(self,pc2_msg):
		points = ros_numpy.point_cloud2.pointcloud2_to_array(pc2_msg)
		return points
	
	def RT_to_Pose_msg(self,RT):
		''' convert RT to geometry_msgs/Pose msg'''
		trans,quat = TfUtils.decompose_tf_M(RT)
		pose_msg = Pose()
		pose_msg.position = list(trans)
		pose_msg.orientation = list(quat)
		return pose_msg
	
	def Pose_msg_to_RT(self,pose_msg):
		''' convert geometry_msgs/Pose msg to RT'''
		RT = TfUtils.compose_tf_M(np.asarray(pose_msg.position),np.asarray(pose_msg.orientation))
	
	def warm_up(self):
		''' warm up the model '''
		self.predictor.init_model()
		fake_points = np.zeros((1, 1024, 3))
		self.predictor.predict(fake_points)
	
if __name__ == '__main__':
	rospy.init_node('pose_prediction_node')
	cfg = get_cfg()
	pred_node = PredictNode(cfg)
	pred_node.warm_up()
	pred_node.start_service()
	rospy.spin()
