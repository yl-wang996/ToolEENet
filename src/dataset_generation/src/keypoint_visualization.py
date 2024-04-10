import os
import struct
import sys

import numpy as np
import open3d as o3d
import rospy
import tf
import yaml
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Header
from utils.transform_utils import TfUtils
from utils.project_utils import project_depth_to_pointscloud, project_xyz_to_pixel_uv
BIT_MOVE_16 = 2 ** 16
BIT_MOVE_8 = 2 ** 8


class KeyPointPublish():
	def __init__(self):
		rospy.init_node('image_publisher', anonymous=True)
		self.pub = rospy.Publisher("key_points", Image, queue_size=2, latch=True)
		self.bridge = CvBridge()
		
	def pub(self, image, sec=0.5):
		try:
			hz = 10
			rate = rospy.Rate(hz)
			for _ in range(int(sec * hz)):
				if not rospy.is_shutdown():
					self.pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
					rate.sleep()
		except rospy.ROSInterruptException:
			print("program interrupted before completion", file=sys.stderr)
			pass


def show_keypoint_in_cam_picture():
	config_yaml = "/homeL/1wang/workspace/toolee_ws/src/dataset_generation/src/cfg/config.yaml"
	with open(config_yaml, 'r') as f:
		cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
	name_map = cfg['ee_name_map']
	
	kp_pub = KeyPointPublish()
	cat = "hammer_grip"
	obj = "hammer_01"
	folder_path = f"/homeL/1wang/workspace/DatasetToolEE_100/{cat}/{obj}"
	pcd_files = [f for f in os.listdir(folder_path) if f.endswith(".pcd")]
	pcd_files.sort()
	ee_poses = {}
	for idx, pcd_file in enumerate(pcd_files):
		
		# load pcd of the object
		pcd = o3d.io.read_point_cloud(os.path.join(folder_path, pcd_file))
		camera_view_matrix = np.loadtxt(os.path.join(folder_path, f"view_matrix_{cat}_{obj}_{idx:04d}.txt"))
		projection_matrix = np.loadtxt(os.path.join(folder_path, f"projection_matrix_{cat}_{obj}_{idx:04d}.txt"))
		for pose_name in name_map[cat]:
			file_name = f"ee_pose_{cat}_{obj}_{pose_name}_{idx:04d}.txt"
			pose_file = os.path.join(folder_path, file_name)
			if os.path.exists(pose_file):
				pose = np.loadtxt(pose_file, delimiter=',')
				# pose = ee_pose_to_cam_view(cam_view_matrx=camera_view_matrix, ee_pose=pose)
				ee_poses[pose_name] = pose
		obj_pose_file_name = os.path.join(folder_path, f"obj_pose_{cat}_{obj}_{idx:04d}.txt")
		obj_pose = np.loadtxt(obj_pose_file_name)
		
		points = np.asarray(pcd.points)
		# points = points_to_cam_view(cam_view_matrx=camera_view_matrix, points=points)
		
if __name__ == '__main__':
	show_keypoint_in_cam_picture()








