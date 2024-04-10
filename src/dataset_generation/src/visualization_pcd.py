# refer: https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/blob/master/lib_cloud_conversion_between_Open3D_and_ROS.py
import os
import struct

import cv2
import numpy as np
import open3d as o3d
import rospy
import tf
import yaml
from PIL import Image as Img
from PIL import ImageDraw
from cv_bridge import CvBridge
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from utils.file_utils import MetaUtils

from utils.transform_utils import TfUtils

BIT_MOVE_16 = 2 ** 16
BIT_MOVE_8 = 2 ** 8


class CamViewPublish():
	def __init__(self, frame_id="map"):
		rospy.init_node("pub_data")
		self.pub_pc = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2, latch=True)
		self.pub_rgb_img = rospy.Publisher("cam_image", Image, queue_size=2, latch=True)
		self.pbu_seg_img = rospy.Publisher("seg_image", Image, queue_size=2, latch=True)
		self.pub_depth_img = rospy.Publisher("depth_image", Image, queue_size=2, latch=True)
		self.bridge = CvBridge()
		self.pub_marker = rospy.Publisher('marker_topic', Marker, queue_size=2)
		self.br = tf.TransformBroadcaster()
		self.fields_xyz = [
			PointField('x', 0, PointField.FLOAT32, 1),
			PointField('y', 4, PointField.FLOAT32, 1),
			PointField('z', 8, PointField.FLOAT32, 1),
		]
		self.fields_xyzrgb = self.fields_xyz + [PointField('rgb', 12, PointField.UINT32, 1)]
		self.packed_points = None
		self.frame_id = frame_id

	def build_marker_msg(self, trans, time_now, marker_id=0, text="ee", scale=0.02):
		marker = Marker()
		marker.header.frame_id = self.frame_id  # Change 'base_link' to your desired frame_id
		marker.header.stamp = time_now
		marker.id = marker_id
		marker.text = text
		marker.type = Marker.SPHERE  # Use SPHERE type for a dot marker
		marker.action = Marker.ADD  # add or modify
		marker.pose.position.x = trans[0]  # Change these values to the desired position
		marker.pose.position.y = trans[1]
		marker.pose.position.z = trans[2]
		marker.pose.orientation.x = 0.0
		marker.pose.orientation.y = 0.0
		marker.pose.orientation.z = 0.0
		marker.pose.orientation.w = 1.0
		marker.scale.x = scale  # Adjust the scale as needed
		marker.scale.y = scale
		marker.scale.z = scale
		marker.color.a = 1.0
		marker.color.r = 1.0  # Set color to red
		marker.color.g = 0.0
		marker.color.b = 0.0
		return marker

	def build_pc2_msg(self, time_now, points: np.ndarray, colors: np.ndarray = None):
		colors = colors * 255
		colors = colors.astype(np.integer)
		packed_points = []
		if colors is None:
			packed_points = points
		else:

			for i in range(len(colors)):
				r, g, b, a = colors[i][0], colors[i][1], colors[i][2], 255
				rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
				x, y, z = points[i][0], points[i][1], points[i][2]
				pt = [x, y, z, rgb]
				packed_points.append(pt)

		header = Header()
		header.frame_id = self.frame_id
		pc2 = point_cloud2.create_cloud(header, self.fields_xyzrgb, packed_points)
		pc2.header.stamp = time_now
		return pc2

	def publish_tf(self, pose_M, child_frame, time, parent_frame="map"):
		trans, quat = TfUtils.decompose_tf_M(pose_M)
		self.br.sendTransform(
			translation=(trans[0], trans[1], trans[2]),
			rotation=quat,
			time=time,
			child=child_frame,
			parent=parent_frame)

	def publish(self, points):
		print("publishing...")

		colors = np.ones((points.shape[0], 3))*255
		colors = colors.astype(np.uint8)
		pc2 = self.build_pc2_msg(points=points,colors=colors, time_now=rospy.Time.now())
		self.pub_pc.publish(pc2)


def get_env_base_pose(env_idx, cfg):
	env_spacing, env_per_row = cfg['env']['env_spacing'], cfg['env']['env_per_row']
	env_base_pos = np.asarray(
		[env_spacing * 2 * (env_idx % env_per_row),
		 env_spacing * 2 * (env_idx // env_per_row),
		 0])
	return env_base_pos


def to_cam_view(cam_view_matrx, points, ee_poses, obj_pose, env_base, cam_transform):
	cam_view_matrx = np.asarray(cam_view_matrx)
	cam_trans, cam_quat = TfUtils.decompose_tf_M(cam_transform)
	obj_trans, obj_quat = TfUtils.decompose_tf_M(obj_pose)
	print(f"cam_trans:{cam_trans}, cam_quat:{cam_quat}")
	print(f"obj_trans:{obj_trans}, obj_quat:{obj_quat}")
	print(TfUtils.decompose_tf_M(cam_view_matrx))
	points += env_base
	ones = np.expand_dims(np.ones(points.shape[0]), axis=-1)
	points = np.concatenate([points, ones], axis=-1)
	points = np.dot(points, cam_view_matrx)
	points = points[:, :3]

	# randmize the ee keypoint
	ee_poses_copy = ee_poses.copy()
	ee_points = {}
	view_pose = camera_pose_to_view_pose(cam_pose=cam_transform)
	# view_pose = cam_transform
	for ee_name, ee_pose in ee_poses_copy.items():
		ee_pose = np.asarray(ee_pose)
		ee_trans, ee_rot = TfUtils.decompose_tf_M(ee_pose)
		ee_point = np.expand_dims(ee_trans + env_base, axis=0)
		ee_point = np.concatenate([ee_point, np.expand_dims(np.ones(ee_point.shape[0]), axis=-1)], axis=-1)
		ee_point = np.dot(ee_point, cam_view_matrx)[:, :3]
		ee_points[ee_name] = ee_point[0]
		# ee_pose[:3, 3] -= obj_trans
		ee_pose_new = np.linalg.inv(view_pose) @ ee_pose
		# ee_pose_new = ee_pose
		# print(f"{TfUtils.decompose_tf_M(ee_pose)} -> {TfUtils.decompose_tf_M(ee_pose_new)}")
		ee_poses[ee_name] = ee_pose_new

	return np.asarray(points), ee_points, ee_poses, view_pose


def to_image_view(image, projection_matrix, ee_points):
	fu = 2 / projection_matrix[0, 0]
	fv = 2 / projection_matrix[1, 1]

	image_copy = image.copy()
	image = np.asarray(image)
	width = image.shape[1]
	height = image.shape[0]
	draw = ImageDraw.Draw(image_copy)

	centerU = width / 2
	centerV = height / 2
	for ee_name, ee_point in ee_points.items():
		x = ee_point[0]
		y = ee_point[1]
		z = ee_point[2]

		u = int(-(x / fu / z * width) + centerU)
		v = int((y / fv / z * height) + centerV)
		r = 5
		draw.ellipse([u - r, v - r, u + r, v + r], fill=(255, 0, 0, 255))
	return image_copy

def camera_pose_to_view_pose(cam_pose):
	view_pose = np.asarray(
		cam_pose @ TfUtils.compose_tf_M(trans=np.asarray([0, 0, 0]), angles=np.asarray([90, 0, -90]) / 180 * np.pi))
	return view_pose


# TODO, now I can show the keypoint in cam view matrix, but not in the right orientation, next to show it in picture

def downsample_points(points, num=1024):
	idx = np.random.choice(points.shape[0], num, replace=False)
	return idx


if __name__ == '__main__':
	cam_pub = CamViewPublish(frame_id="camera")
	cat_name = "wrench"
	wrench = "wrench_"
	ee_name = "head1"
	for i in range(10):
		points = np.load(f"/dataSSD/1wang/dataspace/Dataset3DModel/ee_pcds_norm/{cat_name}/{wrench}{i+1:02d}_{ee_name}.npy")
		cam_pub.publish(
			points=points,
		)
		rospy.sleep(1)

