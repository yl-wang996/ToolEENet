import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
host_name = "tams110"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["ROS_MASTER_URI"] = f"http://{host_name}:11311"
import numpy as np
import rospy
from tool_ee.srv import PosePred
from geometry_msgs.msg import Pose
from utils.transform_utils import TfUtils
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

BIT_MOVE_16 = 2 ** 16
BIT_MOVE_8 = 2 ** 8

fields_xyz = [
	PointField('x', 0, PointField.FLOAT32, 1),
	PointField('y', 4, PointField.FLOAT32, 1),
	PointField('z', 8, PointField.FLOAT32, 1),
]

def points_array_to_pc2_msg(frame_id, points: np.ndarray):
	header = Header()
	header.frame_id = frame_id
	pc2_msg = point_cloud2.create_cloud(header, fields_xyz, points)
	pc2_msg.header.stamp = rospy.Time.now()
	return pc2_msg

def pc2_msg_to_points_array(pc2_msg: PointCloud2):
	return np.array(list(point_cloud2.read_points(pc2_msg, field_names=("x", "y", "z"))))
	
def array_RT_to_msg_Pose(RT):
	''' convert RT to geometry_msgs/Pose msg'''
	trans, quat = TfUtils.decompose_tf_M(RT)
	pose_msg = Pose()
	pose_msg.position = list(trans)
	pose_msg.orientation = list(quat)
	return pose_msg

def msg_Pose_to_array_RT(pose_msg):
	''' convert geometry_msgs/Pose msg to RT'''
	RT = TfUtils.compose_tf_M(np.asarray(pose_msg.position), np.asarray(pose_msg.orientation))
	return RT

def call_pose_pred_service(points: np.ndarray):
	try:
		rospy.wait_for_service('pose_pred', timeout=10)
		rospy.loginfo('pose_pred service is available')
	except rospy.ROSException:
		rospy.logerr('pose_pred service is not available')
		return None
	try:
		pose_pred_service = rospy.ServiceProxy('pose_pred', PosePred)
		pc2_msg = points_array_to_pc2_msg(frame_id='camera_link', points=points)
		responce_msg = pose_pred_service(pc2_msg)
		pred_pose = responce_msg.pred_pose
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
		RT = TfUtils.compose_tf_M(trans=pos, quat=quat)
		angle = TfUtils.quaternion_to_anglexyz(quat)
		rospy.logdebug(f"Predicted result: \n translation: {pos}, \n Predicted angles: {angle}")
		return RT
	except rospy.ServiceException as e:
		print("Service call failed: %s" % e)
	
if __name__ == '__main__':
	rospy.init_node('pose_prediction_node_test',log_level=rospy.DEBUG)
	fake_points = np.zeros((1024, 3))
	call_pose_pred_service(fake_points)