# refer: https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/blob/master/lib_cloud_conversion_between_Open3D_and_ROS.py
import numpy as np
import open3d as o3d
import rospy
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

BIT_MOVE_16 = 2 ** 16
BIT_MOVE_8 = 2 ** 8

fields_xyzrgb = [
    point_cloud2.PointField('x', 0, point_cloud2.PointField.FLOAT32, 1),
    point_cloud2.PointField('y', 4, point_cloud2.PointField.FLOAT32, 1),
    point_cloud2.PointField('z', 8, point_cloud2.PointField.FLOAT32, 1),
    point_cloud2.PointField('rgb', 12, point_cloud2.PointField.UINT32, 1),
]


def o3d_wrap(points, colors):
    colors = colors
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)  # float
    o3d_pcd.colors = o3d.utility.Vector3dVector(colors)  # 0-1, uint8
    return o3d_pcd


def o3d_to_pcd2(o3d_pcd, frame_id):
    points = np.asarray(o3d_pcd.points)
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    colors = np.floor(np.asarray(o3d_pcd.colors) * 255)
    colors = colors[:, 0] * BIT_MOVE_16 + colors[:, 1] * BIT_MOVE_8 + colors[:, 2]
    cloud_data = np.c_[points, colors]
    point_cloud2.create_cloud(header, fields_xyzrgb, cloud_data)
