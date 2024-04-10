import os
import struct
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# TODO: change the host_name to your own host name
host_name = "tams110"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["ROS_MASTER_URI"] = f"http://{host_name}:11311"

import cv2
import tf
from sensor_msgs.msg import Image
import numpy as np
from tool_ee.srv import PosePred
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from utils.file_utils import MetaUtils
from utils.transform_utils import TfUtils
import rospy
from tool_ee.srv import SegPred, SegPredRequest, SegPredResponse
from tool_ee.srv import PosePred, PosePredResponse, PosePredRequest
from cv_bridge import CvBridge

fields_xyz = [
	PointField('x', 0, PointField.FLOAT32, 1),
	PointField('y', 4, PointField.FLOAT32, 1),
	PointField('z', 8, PointField.FLOAT32, 1),
]
fields_xyzrgb = fields_xyz + [PointField('rgb', 12, PointField.UINT32, 1)]
bridge = CvBridge()

cat_ee_map = {
	"hammer_grip": ["head1", "grip"],
	"screwdriver": ["head1", ],
	"wrench": ["head1", ],
}

affordance_seg_id_map={
    "empty": 0,
    "hand": 1,
    "object": 2,
    "hammer_grip_head1": 3,
    "hammer_grip_grip": 4,
    "screwdriver_head1": 5,
    "wrench_head1": 6,
    "wrench_head2": 7,
}


def apply_pose_prior(pose_prior, pred_obj_RT):
	# rescale the ee pose prior according to the domain randomization
	refined_pred_RT = np.asarray(pred_obj_RT) @ np.asarray(pose_prior)
	return refined_pred_RT


def load_pose_prior(cat_name, ee_name):
	default_scale_map = {
		"hammer_grip": 0.3,
		"screwdriver": 0.2,
		"wrench": 0.2,
	}
	scale = default_scale_map[cat_name]
	data_path = "/dataSSD/yunlong/dataspace/Dataset3DModel"
	pose_path_list = [f for f in os.listdir(os.path.join(data_path, cat_name)) if
	                  f.endswith('_pose.txt') and ee_name in f]
	pose_all = []
	for pose_path in pose_path_list:
		pose = np.loadtxt(os.path.join(data_path, cat_name, pose_path), delimiter=',', dtype=float)
		pose_all.append(pose)
	pose_all = np.expand_dims(np.array(pose_all), axis=0)
	pose_prior_RT = TfUtils.get_avg_sRT(pose_all)[0]
	pose_prior_RT[:3, 3] = pose_prior_RT[:3, 3] * scale
	return pose_prior_RT


def call_pose_pred_service(points: np.ndarray, task_type="ee_pose", symtrs_threshold=0.5):
	"""
		fake_points = np.zeros((1024, 3))
		call_pose_pred_service(fake_points)
		
	Args:
		points:
	Returns:
	"""
	
	print(f"calling pose_pred service, task_type:{task_type}...")
	
	try:
		rospy.wait_for_service('pose_pred', timeout=10)
		rospy.loginfo('pose_pred service is available')
	except rospy.ROSException:
		rospy.logerr('pose_pred service is not available')
		rospy.signal_shutdown('pose_pred service not available')
	try:
		if points.shape[0] > 1024:
			points, _ = sample_data(points, num_sample=1024)
		pose_pred_service = rospy.ServiceProxy('pose_pred', PosePred)
		# build the pointcloud2 message
		header = Header()
		header.frame_id = 'camera_link'
		pc2_msg = point_cloud2.create_cloud(header, fields_xyz, points)
		pc2_msg.header.stamp = rospy.Time.now()
		request_msg = PosePredRequest(pc2=pc2_msg)
		# call the service
		responce_msg = pose_pred_service(request_msg)
		# parse the responce message
		pred_pose = responce_msg.pred_pose
		symtrs = responce_msg.pred_symtrs
		pos = np.asarray([
			pred_pose.position.x,
			pred_pose.position.y,
			pred_pose.position.z
		])
		quat = np.asarray([
			pred_pose.orientation.x,
			pred_pose.orientation.y,
			pred_pose.orientation.z,
			pred_pose.orientation.w
		])
		# convert the pose to the RT matrix
		RT = TfUtils.compose_tf_M(trans=pos, quat=quat)
		angle = TfUtils.quaternion_to_anglexyz(quat)
		symtrs = np.asarray(symtrs)
		symtrs[symtrs < symtrs_threshold] = 0
		symtrs[symtrs > symtrs_threshold] = 1
		rospy.logdebug(f"Predicted result: \n translation: {pos}, \n Predicted angles: {angle}, symmetry: {symtrs}")
		if task_type == "ee_pose":
			return RT, symtrs
		elif task_type == "obj_pose":
			return RT
		else:
			raise ValueError(f"task_type: {task_type} is not supported.")
	except rospy.ServiceException as e:
		print("Service call failed: %s" % e)
		return None, None


def call_seg_pred_service(rgb_img: np.ndarray, vis=True):
	try:
		rospy.wait_for_service('seg_pred', timeout=5)
		rospy.loginfo('seg_pred service is available')
	except rospy.ROSException:
		rospy.logerr('seg_pred service is not available')
		rospy.signal_shutdown('seg_pred service not available')
	try:
		seg_pred_service = rospy.ServiceProxy('seg_pred', SegPred)
		
		request_msg = SegPredRequest(rgb=bridge.cv2_to_imgmsg(rgb_img, encoding='rgb8'), vis=vis)
		response_meg = seg_pred_service(request_msg)
		seg_msg_list = response_meg.seg_list
		seg_name_list = response_meg.seg_name_list
		seg_masks = []
		for idx, _ in enumerate(seg_name_list):
			img_msg = seg_msg_list[idx]
			seg_img = bridge.imgmsg_to_cv2(img_msg)
			seg_mask = np.zeros_like(seg_img, dtype=bool)
			seg_mask[seg_img == 1] = True
			seg_mask[seg_img == 0] = False
			seg_masks.append(seg_mask)
		return seg_masks, seg_name_list
	
	except rospy.ServiceException as e:
		print("Service call failed: %s" % e)
		rospy.signal_shutdown('seg_pred service not available')


def sample_data(data, num_sample=1024):
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


class Visualization():
	def __init__(self, frame_id="camera_link"):
		self.pub_pc = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2, latch=True)
		self.pub_rgb_img = rospy.Publisher("cam_image", Image, queue_size=2, latch=True)
		self.pub_depth_img = rospy.Publisher("depth_image", Image, queue_size=2, latch=True)
		self.pub_seg_gt_img = rospy.Publisher("seg_gt_image", Image, queue_size=2, latch=True)
		self.bridge = CvBridge()
		self.br = tf.TransformBroadcaster()
		
		self.packed_points = None
		self.frame_id = frame_id
	
	def build_pc2_msg(self, time_now, points: np.ndarray, colors: np.ndarray = None):
		packed_points = []
		if colors is None:
			packed_points = points
		else:
			# colors = colors * 255
			colors = colors.astype(np.integer)
			for i in range(len(colors)):
				r, g, b, a = colors[i][0], colors[i][1], colors[i][2], 255
				rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
				x, y, z = points[i][0], points[i][1], points[i][2]
				pt = [x, y, z, rgb]
				packed_points.append(pt)
		
		header = Header()
		header.frame_id = self.frame_id
		pc2 = point_cloud2.create_cloud(header, fields_xyzrgb, packed_points)
		pc2.header.stamp = time_now
		return pc2
	
	def publish_tf(self, pose_M, child_frame, time, parent_frame="camera_link"):
		trans, quat = TfUtils.decompose_tf_M(pose_M)
		self.br.sendTransform(
			translation=(trans[0], trans[1], trans[2]),
			rotation=quat,
			time=time,
			child=child_frame,
			parent=parent_frame)
	
	def publish(self, image, points, colors, ee_poses_pred, ee_poses_gt, obj_pose, depth_img,seg_gt_img, sec=0.5):
		print("publishing...")
		hz = 10
		rate = rospy.Rate(hz)
		# publish the rgb image
		self.pub_rgb_img.publish(self.bridge.cv2_to_imgmsg(image, 'rgb8'))
		
		# publish the depth image
		self.pub_depth_img.publish(self.bridge.cv2_to_imgmsg(depth_img))
		
		# publish the seg_gt image
		self.pub_seg_gt_img.publish(self.bridge.cv2_to_imgmsg(seg_gt_img))
		
		# publish the point cloud for hand and object
		pc2 = self.build_pc2_msg(points=points, colors=colors, time_now=rospy.Time.now())
		self.pub_pc.publish(pc2)
		
		for _ in range(int(sec * hz)):
			time_now = rospy.Time.now()
			for ee_name, ee_pose in ee_poses_pred.items():
				label = ee_name.split("_")[-1]
				self.publish_tf(pose_M=ee_pose, child_frame=label + "pred", parent_frame=self.frame_id, time=time_now)
				self.publish_tf(pose_M=ee_poses_gt[label], child_frame=label + "gt", parent_frame=self.frame_id,
				                time=time_now)
			if obj_pose is not None:
				self.publish_tf(pose_M=obj_pose, child_frame="obj", parent_frame=self.frame_id, time=time_now)
			rate.sleep()


def depth_to_img(obj_depth):
	obj_depth_img = obj_depth.copy()
	# nromalzie the depth to 0-1
	depth_min = np.min(np.unique(obj_depth_img)[1:])
	obj_depth_img[obj_depth_img == np.unique(obj_depth_img)[0]] = depth_min
	obj_depth_img = (obj_depth_img - np.min(obj_depth_img)) / (np.max(obj_depth_img) - np.min(obj_depth_img))
	# convert the depth to the rgb image
	obj_depth_img = (obj_depth_img * 255).astype(np.uint8)
	depth_img = np.repeat(obj_depth_img[:, :, np.newaxis], 3, axis=2)
	return depth_img


def depth_to_point_cloud(depth_array, camera_proj_matrix, width, height, seg_mask=None, rgb_img=None):
	fu = 2 / camera_proj_matrix[0, 0]
	fv = 2 / camera_proj_matrix[1, 1]
	centerU = width / 2
	centerV = height / 2
	
	u = range(0, width)
	v = range(0, height)
	
	u, v = np.meshgrid(u, v)
	u = u.astype(float)
	v = v.astype(float)
	
	Z = depth_array
	X = -(u - centerU) / width * Z * fu
	Y = (v - centerV) / height * Z * fv
	
	Z = Z.flatten()
	# TODO: check the depth threshold value if you change the data
	depth_valid = Z > -10001
	
	if seg_mask is not None:
		valid = np.logical_and(depth_valid, seg_mask.flatten())
	else:
		valid = depth_valid
	
	X = X.flatten()
	Y = Y.flatten()
	
	position = np.vstack((X, Y, Z, np.ones(len(X))))[:, valid].T
	points = position[:, 0:3]
	if rgb_img is not None:
		colors = rgb_img.reshape((-1, 3))[valid]
		return np.asarray(points), np.asarray(colors)
	else:
		return np.asarray(points), None


def ee_centric_estimate(rgb_img, depth_array, camera_proj_matrix, cat_name, task_type, vis_mask=True):
	ee_poses = {}
	min_points_num = 20
	seg_masks, seg_name_list = call_seg_pred_service(rgb_img=rgb_img, vis=vis_mask)
	ee_masks = {}
	ee_points = {}
	ee_colors = {}
	ee_symtrs = []
	for idx, seg_mask in enumerate(seg_masks):
		if True not in seg_mask:
			print(f"seg_name: {seg_name_list[idx]} has no valid segmentations, discard it.")
			continue
		affordance_name = seg_name_list[idx]
		ee_name = affordance_name.split("_")[-1]
		if ee_name not in cat_ee_map[cat_name]:
			print(f"seg_name: {affordance_name} is not a valid ee name for {cat_name}, discard it.")
			continue
		# get the point cloud
		points, colors = depth_to_point_cloud(
			depth_array=depth_array,
			seg_mask=seg_mask,
			camera_proj_matrix=camera_proj_matrix,
			width=rgb_img.shape[1],
			height=rgb_img.shape[0],
			rgb_img=rgb_img
		)
		print(f"seg_name: {affordance_name} has {points.shape[0]} points.")
		if points.shape[0] < min_points_num:
			rospy.loginfo(f"seg_name: {affordance_name} has less than {min_points_num} points, discard it.")
			continue
		# call the pose prediction service
		ee_pose, ee_symtr = call_pose_pred_service(points=points, task_type=task_type)
		if ee_pose is None:
			continue
		ee_poses[affordance_name] = ee_pose
		ee_masks[affordance_name] = seg_mask
		points, sample_idx = sample_data(points, num_sample=1024)
		colors = colors[sample_idx]
		ee_points[affordance_name] = points
		ee_colors[affordance_name] = colors
		ee_symtrs.append(ee_symtr)
	return ee_poses, ee_symtrs, ee_masks, ee_points, ee_colors


def obj_centric_estimate(obj_points, cat_name, task_type):
	ee_poses = {}
	ee_names = cat_ee_map[cat_name]
	obj_pose = call_pose_pred_service(points=obj_points, task_type=task_type)
	for ee_name in ee_names:
		pose_prior_RT = load_pose_prior(cat_name, ee_name)
		ee_pose = apply_pose_prior(pose_prior=pose_prior_RT, pred_obj_RT=obj_pose)
		ee_poses[ee_name] = ee_pose
	return ee_poses, obj_pose


def get_refine_angle(R_pred, R_gt, symtr_axis):
	"""
	Args:
		RT_pred: [3,3]
		RT_gt:  [3,3]
	Returns:
	"""
	stmtry_axes_map = {
		0: 'rxyz',
		1: 'ryzx',
		2: 'rzyx',
	}
	R1 = R_pred / np.cbrt(np.linalg.det(R_pred))
	R2 = R_gt / np.cbrt(np.linalg.det(R_gt))
	R = R1 @ R2.transpose()
	RT = np.eye(4)
	RT[:3, :3] = R
	_angle = np.asarray(TfUtils.tf_M_to_anglexyz(RT, axes=stmtry_axes_map[symtr_axis]))
	return _angle[0]


# TODO: fine tuning this function with using the real hand pose, even it is the same currently
def pose_refinement(RT_pred, RT_gt, symtr, cat_name):
	"""
	
	Args:
		RT_pred: [4,4]
		RT_gt:  [4,4]
		symtr: [3] xyz
		cat_name: str

	Returns:
	"""
	RT_pred = np.asarray(RT_pred)
	RT_gt = np.asarray(RT_gt)
	if cat_name == "wrench":
		# TODO, fix the symtr estimation for wrench
		symtr = np.array([0, 1, 0])
	
	symtr = np.asarray(symtr)
	if 1 not in symtr:
		return RT_pred
	
	sym_axis_list = np.where(symtr == 1)[0]
	assert len(sym_axis_list) == 1, "only support single axis symmetry"
	sym_axis = sym_axis_list[0]
	
	_angle = np.zeros(3)
	_angle[sym_axis] = get_refine_angle(RT_pred[:3, :3], RT_gt[:3, :3], symtr_axis=sym_axis)
	_RT = TfUtils.compose_tf_M(trans=np.zeros(3), angles=_angle)
	RT_pred_refined = RT_pred @ _RT
	return RT_pred_refined


def test():
	# TODO: find all the symmetric example and refine it
	# TODO: all the example in figure
	"""
		test_set_data example:
			symtry:
				wrench_01: 503
				screwdriver_01: 508
			asymtry:
				hammer_01: 500
		novel_set_data example:
			symtry:
				wrench_18: 517
				screwdriver_15: 500
			asymtry:
				hammer_14: 500

	"""
	
	rospy.init_node("visual_node")
	cam_pub = Visualization(frame_id="camera_link")
	task_type = "obj_pose"  # "ee_pose" or "obj_pose"
	refine_pose = False  # True or False
	cat_name = "wrench"  # hammer_grip, screwdriver, wrench
	# obj in test set
	# obj_name = "wrench_01"  # hammer_01, screwdriver_01, wrench_01
	# obj in novel set
	obj_name = "wrench_01"  # hammer_14, screwdriver_15, wrench_18
	data_root = f"/dataSSD/yunlong/dataspace/"
	toolee_data_root = os.path.join(data_root, "DatasetToolEE")
	meta_list = np.loadtxt(os.path.join(toolee_data_root, f'val_examples_ee_visible.txt'), delimiter=',', dtype=str)
	if task_type == "obj_pose":
		assert refine_pose == False, "refine_pose must be False for obj_pose"
	ee_poses = {}
	# for idx in range(500,600):
	for idx in [503]:
		print(f"idx: {idx}")
		meta_file_name = f"meta_{cat_name}_{obj_name}_{idx:04d}.yaml"
		print(meta_file_name)
		meta_name = os.path.join(toolee_data_root, cat_name, obj_name, meta_file_name)
		meta_util = MetaUtils(data_root=toolee_data_root, meta_name=meta_name)
		cat_name, obj_name, example_id = meta_util.get_cat_obj_id()
		# load the depth array
		obj_depth_array = meta_util.get_depth_array()
		depth_img = depth_to_img(obj_depth_array)
		# load rgb image
		rgb_img = meta_util.get_image().astype(np.uint8)
		# load the camera intrinsic matrix
		projection_matrix = meta_util.get_cam_proj_matrix()
		h, w = meta_util.get_cam_hw()
		all_points_cloud, all_colors = depth_to_point_cloud(
			depth_array=obj_depth_array,
			camera_proj_matrix=projection_matrix,
			width=w,
			height=h,
			seg_mask=None,
			rgb_img=rgb_img
		)
		all_points_cloud, idx = sample_data(all_points_cloud, num_sample=10240)
		if all_colors is not None:
			all_colors = all_colors[idx]
		
		obj_points, obj_colors = meta_util.get_obj_point_cloud()
		obj_pose = None
		# estimate the end effector poses from RGB-D image
		if task_type == "ee_pose":
			ee_poses_pred, ee_symtrs, ee_masks, ee_points, ee_colors = ee_centric_estimate(
				rgb_img=rgb_img,
				depth_array=obj_depth_array,
				camera_proj_matrix=projection_matrix,
				cat_name=cat_name,
				task_type=task_type,
				vis_mask=True)
		elif task_type == "obj_pose":
			ee_poses_pred, obj_pose = obj_centric_estimate(obj_points=obj_points, cat_name=cat_name,
			                                               task_type=task_type)
		else:
			raise ValueError(f"task_type: {task_type} is not supported.")
		
		ee_poses_gt = meta_util.get_ee_poses()
		if refine_pose:
			for idx, iterm in enumerate(ee_poses_pred.items()):
				affordance_name, ee_pose_pred = iterm
				ee_name = affordance_name.split("_")[-1]
				refined_pose = pose_refinement(RT_pred=ee_pose_pred, RT_gt=ee_poses_gt[ee_name], symtr=ee_symtrs[idx],
				                               cat_name=cat_name)
				ee_poses_pred[affordance_name] = refined_pose
		cam_pub.publish(
			image=rgb_img,
			points=all_points_cloud,
			colors=all_colors,
			ee_poses_pred=ee_poses_pred,
			ee_poses_gt=ee_poses_gt,
			obj_pose=obj_pose,
			depth_img=depth_img,
			sec=0.5,
		)


if __name__ == '__main__':
	"""
	hammer_grip:
		hammer_01: [508, 519, 517]
		hammer_08: [500, 503, 507]
		hammer_14: [500, 502, 503]
	screwdriver:
		screwdriver_01: [500, 505, 507]
		screwdriver_04: [501, 502, 505]
		screwdriver_15: [501, 502, 503]
	wrench:
		wrench_01: [503, 504, 505]
		wrench_15: [509, 510, 515]
		wrench_18: [500, 501, 507]
	"""
	
	rospy.init_node("visual_node")
	cam_pub = Visualization(frame_id="camera_link")
	task_type = "obj_pose"  # "ee_pose" or "obj_pose"
	refine_pose = False  # True or False
	cat_name = "wrench"  # hammer_grip, screwdriver, wrench
	# obj in test set
	obj_name = "wrench_18"
	# obj in novel set "wrench_01"  # hammer_14, screwdriver_15, wrench_18
	ids = [507]
	data_root = f"/dataSSD/yunlong/dataspace/"
	toolee_data_root = os.path.join(data_root, "DatasetToolEE")
	meta_list = np.loadtxt(os.path.join(toolee_data_root, f'val_examples_ee_visible.txt'), delimiter=',', dtype=str)
	if task_type == "obj_pose":
		assert refine_pose == False, "refine_pose must be False for obj_pose"
	ee_poses = {}
	# for idx in range(500, 600):
	for idx in ids:
		print(f"idx: {idx}")
		meta_file_name = f"meta_{cat_name}_{obj_name}_{idx:04d}.yaml"
		print(meta_file_name)
		meta_name = os.path.join(toolee_data_root, cat_name, obj_name, meta_file_name)
		meta_util = MetaUtils(data_root=toolee_data_root, meta_name=meta_name)
		cat_name, obj_name, example_id = meta_util.get_cat_obj_id()

		# load the depth array
		obj_depth_array = meta_util.get_depth_array()
		depth_img = depth_to_img(obj_depth_array)
		# load rgb image
		rgb_img = meta_util.get_image().astype(np.uint8)
		# load the camera intrinsic matrix
		projection_matrix = meta_util.get_cam_proj_matrix()
		h, w = meta_util.get_cam_hw()
		all_points_cloud, all_colors = depth_to_point_cloud(
			depth_array=obj_depth_array,
			camera_proj_matrix=projection_matrix,
			width=w,
			height=h,
			seg_mask=None,
			rgb_img=rgb_img
		)
		all_points_cloud, idx = sample_data(all_points_cloud, num_sample=int(10240 / 2))
		if all_colors is not None:
			all_colors = all_colors[idx]
		
		obj_points, obj_colors = meta_util.get_obj_point_cloud()
		obj_pose = None
		# estimate the end effector poses from RGB-D image
		if task_type == "ee_pose":
			ee_poses_pred, ee_symtrs, ee_masks, ee_points, ee_colors = ee_centric_estimate(
				rgb_img=rgb_img,
				depth_array=obj_depth_array,
				camera_proj_matrix=projection_matrix,
				cat_name=cat_name,
				task_type=task_type,
				vis_mask=True)
		elif task_type == "obj_pose":
			ee_poses_pred, obj_pose = obj_centric_estimate(obj_points=obj_points, cat_name=cat_name, task_type=task_type)
		else:
			raise ValueError(f"task_type: {task_type} is not supported.")
		
		ee_poses_gt = meta_util.get_ee_poses()
		if refine_pose:
			for idx, iterm in enumerate(ee_poses_pred.items()):
				affordance_name, ee_pose_pred = iterm
				ee_name = affordance_name.split("_")[-1]
				refined_pose = pose_refinement(RT_pred=ee_pose_pred, RT_gt=ee_poses_gt[ee_name], symtr=ee_symtrs[idx],
				                               cat_name=cat_name)
				ee_poses_pred[affordance_name] = refined_pose
				
		seg_gt = meta_util.get_seg()
		seg_gt_img = np.zeros_like(rgb_img)
		for seg_id in np.unique(seg_gt):
			if seg_id == 0:
				continue
			rnd_color = np.random.randint(0, 255, 3)
			seg_gt_img[seg_gt == seg_id] = rnd_color

		cam_pub.publish(
			image=rgb_img,
			points=all_points_cloud,
			colors=all_colors,
			ee_poses_pred=ee_poses_pred,
			ee_poses_gt=ee_poses_gt,
			obj_pose=obj_pose,
			depth_img=depth_img,
			seg_gt_img=seg_gt_img,
			sec=1,
		)
