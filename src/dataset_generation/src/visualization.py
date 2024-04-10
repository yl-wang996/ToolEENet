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


def camera_pose_to_view_pose(cam_pose):
    view_pose = np.asarray(cam_pose @ TfUtils.compose_tf_M(trans=np.asarray([0, 0, 0]), angles=np.asarray([90, 0, -90]) / 180 * np.pi))
    return view_pose

class CamViewPublish():
    def __init__(self, frame_id="map"):
        rospy.init_node("pub_data")
        self.pub_pc = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2, latch=True)
        self.pub_color_img = rospy.Publisher("cam_image", Image, queue_size=2, latch=True)
        self.pub_affordance_seg = rospy.Publisher("affordance_seg", Image, queue_size=2, latch=True)
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
    
    def build_marker_msg(self, trans, time_now, marker_id=0, text="ee", scale=0.02 ):
        marker = Marker()
        marker.header.frame_id = self.frame_id  # Change 'base_link' to your desired frame_id
        marker.header.stamp = time_now
        marker.id = marker_id
        marker.text = text
        marker.type = Marker.SPHERE  # Use SPHERE type for a dot marker
        marker.action = Marker.ADD # add or modify
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

    def publish_tf(self, pose_M, child_frame, time, parent_frame="map" ):
        trans, quat = TfUtils.decompose_tf_M(pose_M)
        self.br.sendTransform(
            translation=(trans[0], trans[1], trans[2]),
            rotation=quat,
            time=time,
            child=child_frame,
            parent=parent_frame)

    def seg_to_image(self, seg):
        seg_image = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        seg_ids = np.unique(seg)
        color = np.random.randint(0, 255, (len(seg_ids), 3))
        for i, seg_id in enumerate(seg_ids):
            seg_image[seg == seg_id] = color[i]
        return seg_image


    def publish(self, image, affordance_seg, points, colors, ee_points, obj_pose, ee_poses, cam_pose, sec=0.5):
        print("publishing...")
        hz = 10
        rate = rospy.Rate(hz)
        cv_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        for ee_name, ee_point in ee_points.items():
            cv2.circle(cv_image, (ee_point[0],ee_point[1]), 5, (255,0,0), 5)
        color_image_msg = self.bridge.cv2_to_imgmsg(cv_image)
        affordance_seg = self.seg_to_image(affordance_seg)
        affordance_seg_msg = self.bridge.cv2_to_imgmsg(affordance_seg)

        for _ in range(sec*hz):
            time_now = rospy.Time.now()
            pc2 = self.build_pc2_msg(points=points, colors=colors,time_now=time_now)
            self.pub_pc.publish(pc2)
            self.pub_color_img.publish(color_image_msg)
            self.pub_affordance_seg.publish(affordance_seg_msg)

            for ee_name, ee_pose in ee_poses.items():
                self.publish_tf(pose_M=ee_pose, child_frame=ee_name, parent_frame=self.frame_id, time=time_now)
            self.publish_tf(pose_M=cam_pose, child_frame="camera", parent_frame=self.frame_id, time=time_now)
            self.publish_tf(pose_M=obj_pose, child_frame="object", parent_frame=self.frame_id, time=time_now)
            rate.sleep()

if __name__ == '__main__':
    config_yaml = "/homeL/1wang/workspace/toolee_ws/src/dataset_generation/src/cfg/config.yaml"
    with open(config_yaml, 'r') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    ee_names = cfg['ee_name_map']

    cam_pub = CamViewPublish(frame_id="map")
    cat_name = "hammer_grip"
    obj_name = "hammer_01"
    data_root = f"/dataSSD/1wang/dataspace/DatasetToolEE/"

    for idx in range(10):
        meta_file_name = f"meta_{cat_name}_{obj_name}_{idx:04d}.yaml"
        meta_util = MetaUtils(data_root, os.path.join(cat_name, obj_name, meta_file_name))
        image = meta_util.get_image()
        points, colors = meta_util.get_obj_point_cloud()
        ee_points = meta_util.get_ee_point()
        ee_poses = meta_util.get_ee_poses()
        cam_transform = meta_util.get_cam_tranform()
        obj_pose=meta_util.get_obj_pose()
        affordance_seg = meta_util.get_affordance_seg()
        cam_pub.publish(
            image=np.asarray(image).astype(np.uint8),
            affordance_seg=affordance_seg,
            points=np.asarray(points),
            colors=np.asarray(colors),
            obj_pose=obj_pose,
            ee_points=ee_points,
            ee_poses=ee_poses,
            cam_pose=cam_transform,
            sec=5,
        )