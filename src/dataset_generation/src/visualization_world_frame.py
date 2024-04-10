# refer: https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/blob/master/lib_cloud_conversion_between_Open3D_and_ROS.py
import os
import struct

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
BIT_MOVE_16 = 2 ** 16
BIT_MOVE_8 = 2 ** 8


class PCDPublish():
    def __init__(self, frame_id="base_link"):
        rospy.init_node("pub_data")
        self.pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2, latch=True)
        self.br = tf.TransformBroadcaster()
        self.fields_xyz = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        self.fields_xyzrgb = self.fields_xyz + [PointField('rgb', 12, PointField.UINT32, 1)]
        self.packed_points = None
        self.frame_id = frame_id

    def pc2_wrap(self, packed_points):
        header = Header()
        header.frame_id = self.frame_id
        pc2 = point_cloud2.create_cloud(header, self.fields_xyzrgb, packed_points)
        pc2.header.stamp = rospy.Time.now()
        return pc2


    def publish(self, points, colors,obj_pose, ee_poses,sec=0.5):
        print("publishing points")

        hz = 10
        rate = rospy.Rate(hz)
        packed_points = self.point_pack(np.asarray(points), np.asarray(colors))
        pc2 = self.pc2_wrap(packed_points)
        
        # trans = tf.transformations.translation_from_matrix(obj_pose)
        # obj_trans = np.asarray([1,1,-1])
        obj_trans, obj_rot = TfUtils.decompose_tf_M(obj_pose)
        # obj_rot = TfUtils.anglexyz_to_quaternion(np.array([0,0,0]))
        print(f"obj_pose:{obj_trans}, {obj_rot}")
        
        for n in range(int(sec * hz)):
            if not rospy.is_shutdown():
                self.pub.publish(pc2)
                for key, ee_pose in ee_poses.items():
                    ee_trans, ee_rot = TfUtils.decompose_tf_M(ee_pose)
                    self.br.sendTransform(
                        ee_trans,
                        ee_rot,
                        rospy.Time.now(),
                        child=f"{key}_pose",
                        parent="map")
                    
                self.br.sendTransform(
                    obj_trans,
                    obj_rot,
                    rospy.Time.now(),
                    child=f"obj_pose",
                    parent="map")
                rate.sleep()

    def point_pack(self, points: np.ndarray, colors: np.ndarray = None):
        colors = colors*255
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
        return packed_points

    def fake_pcd(self):
        packed_points = []
        lim = 800
        for i in range(lim):
            for j in range(lim):
                for k in range(lim):
                    x = float(i) / 100
                    y = float(j) / 100
                    z = float(k) / 100
                    r = int(x / 8 * 255.0)
                    g = int(y / 8 * 255.0)
                    b = int(z / 8 * 255.0)
                    a = 255
                    # print(r, g, b, a)
                    rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                    # print(hex(rgb))
                    pt = [x, y, z, rgb]
                    packed_points.append(pt)
        return packed_points

# def ee_pose_to_cam_view(cam_view_matrx, ee_pose):
#     ee_pose = np.matrix(ee_pose)
#     cam_view_matrx = np.matrix(cam_view_matrx)
#     ee_pose = ee_pose*cam_view_matrx
#     return ee_pose



# def points_to_cam_view(cam_view_matrx, points):
#     cam_view_matrx = np.matrix(cam_view_matrx)
#     points = np.concatenate([points, np.expand_dims(np.ones(points.shape[0]), axis=-1)], axis=-1)
#     points = points*cam_view_matrx  # when the data type is mp.matrix, then the * means dot production
#     points = points[..., :3]
#     return np.asarray(points)

def random_transform(points, ee_poses, obj_pose):
    obj_trans = obj_pose[:3,3].T
    rnd_M = TfUtils.random_tf_M()
    # randomize the points cloud
    points -= obj_trans
    ones = np.expand_dims(np.ones(points.shape[0]), axis=-1)
    points = np.concatenate([points, ones], axis=-1)
    points = np.dot(rnd_M, points.T)
    points = points.T[:,:3]
    points += obj_trans
    
    # randmize the ee poses
    ee_poses_copy = ee_poses.copy()
    for ee_name, ee_pose in ee_poses_copy.items():
        ee_pose = ee_poses[ee_name]
        ee_pose[:3,3] -= obj_pose[:3,3]
        ee_pose = np.dot(rnd_M, ee_pose)
        ee_pose[:3,3] += obj_pose[:3,3]
        ee_poses[ee_name] = np.asarray(ee_pose)
    
    # randomize the obj pose
    new_obj_pose = obj_pose.copy()
    new_obj_pose[:3,3] -= obj_pose[:3,3]
    new_obj_pose = np.dot(rnd_M, new_obj_pose)
    new_obj_pose[:3,3] += obj_pose[:3,3]

    return np.asarray(points), ee_poses, new_obj_pose

def get_env_base_pose(env_idx, cfg):
    env_spacing, env_per_row = cfg['env']['env_spacing'], cfg['env']['env_per_row']
    env_base_pos = np.asarray(
        [env_spacing * 2 * (env_idx % env_per_row),
         env_spacing * 2 * (env_idx // env_per_row),
         0])
    return env_base_pos

def show_in_world_frame(rnd=False, sec=2):
    is_rnd = rnd
    config_yaml = "/homeL/1wang/workspace/toolee_ws/src/dataset_generation/src/cfg/config.yaml"
    with open(config_yaml, 'r') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    name_map = cfg['ee_name_map']
    
    # pcd_pub = PCDPublish()
    # packed_points = pcd_pub.fake_pcd()
    # pcd_pub.publish(packed_points, [0, 0, 0, 0, 0, 0, 1])
    
    pcd_pub = PCDPublish(frame_id="map")
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
        if is_rnd:
            points, ee_poses, obj_pose = random_transform(points, ee_poses, obj_pose)
        pcd_pub.publish(
            points=points,
            colors=pcd.colors,
            obj_pose=obj_pose,
            ee_poses=ee_poses,
            sec=sec,
        )

if __name__ == '__main__':
    show_in_world_frame(rnd=False, sec=1)