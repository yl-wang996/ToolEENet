
import rospy
import copy
import os

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import Marker,InteractiveMarker,InteractiveMarkerControl
from geometry_msgs.msg import Point
from tf.broadcaster import TransformBroadcaster
import numpy as np
import tf
from tf.transformations import compose_matrix, decompose_matrix

from random import random
from math import sin

#####################################################################
# Marker Creation

def normalizeQuaternion( quaternion_msg ):
    norm = quaternion_msg.x**2 + quaternion_msg.y**2 + quaternion_msg.z**2 + quaternion_msg.w**2
    s = norm**(-0.5)
    quaternion_msg.x *= s
    quaternion_msg.y *= s
    quaternion_msg.z *= s
    quaternion_msg.w *= s

def compose_tf_M(trans, angles, scale):
    shear = np.array([0, 0, 0])
    persp = np.array([0, 0, 0, 1])
    M = compose_matrix(np.asarray(scale), shear, np.asarray(angles), np.asarray(trans), persp)
    return M

def decompose_tf_M(M):
    scale, shear, angles, trans, persp = decompose_matrix(M)
    return np.asarray(trans), np.asarray(angles), np.asarray(scale)

class Interactive6DMeshMarker():
    def __init__(self,frame_id="base_link"):
        self.frame_id = frame_id
        self.server = InteractiveMarkerServer("interactive_marker_controls")
        self.counter = 0
        self.menu_id_pose_name = {
            4: "head1",
            5: "grip",
            7: "head1",
            9: "head1",
            10: "head2",
        }
        self.pose_pub_sec = 5
        self.pose_pub_rate = 3
        self.marker_coord_size = 0.7
        self.mesh_size = 1

    def processFeedback(self, feedback:InteractiveMarkerFeedback):
        menu_entry_id = feedback.menu_entry_id
        pose_name = self.menu_id_pose_name[menu_entry_id]
        # tf.transformations.
        # tf.transformations.quaternion_matrix([feedback.pose.orientation.x, feedback.pose.orientation.y, feedback.pose.orientation.z, feedback.pose.orientation.w])
        trans = np.array([feedback.pose.position.x, feedback.pose.position.y, feedback.pose.position.z])
        angles = tf.transformations.euler_from_quaternion([feedback.pose.orientation.x, feedback.pose.orientation.y, feedback.pose.orientation.z, feedback.pose.orientation.w])
        scale = np.array([1, 1, 1])
        M = compose_tf_M(
            trans=trans,
            angles=angles,
            scale=scale)
        M = np.linalg.inv(M) # inverse the pose of the object to get the pose of the end-effector
        print(M)
        save_path = os.path.join(data_root,self.mesh_name.replace(".stl",f"_{pose_name}_pose.txt"))
        print(f"save to :{save_path}")
        np.savetxt(save_path, M, delimiter=',')
    
    def showposeCallback(self, feedback):
        files = os.listdir(os.path.join(data_root, new_dataset_name, cat_name))
        pose_files = [f for f in files if f.endswith("_pose.txt") and f.startswith(obj_name)]
        trans_list = []
        quat_list = []
        pose_label_list = []
        for pose_file in pose_files:
            pose_path = os.path.join(data_root, new_dataset_name, cat_name, pose_file)
            M = np.loadtxt(pose_path, delimiter=',')
            trans, angle, scale = decompose_tf_M(M)
            quat = tf.transformations.quaternion_from_euler(angle[0], angle[1], angle[2])
            trans_list.append(trans)
            quat_list.append(quat)
            pose_label_list.append(pose_file.split('.')[0])
        tf_broadcaster = tf.TransformBroadcaster()
        rate = rospy.Rate(self.pose_pub_rate)
        for _ in range(int(self.pose_pub_sec*self.pose_pub_rate)):
            for pose_id in range(len(pose_files)):
                tf_broadcaster.sendTransform(
                    trans_list[pose_id],
                    quat_list[pose_id],
                    rospy.Time.now(),
                    parent="map",
                    child=f"{pose_label_list[pose_id]}",)
            rate.sleep()
        
    def emptyCallback(self, feedback):
        pass

    # create the mesh of the inactivate marker
    def makeMesh(self, mesh_name):
        self.mesh_name = mesh_name  # cat/obj.stl
        marker = Marker()
        marker.type = Marker.MESH_RESOURCE
        mesh_file_path = f"package://pose_annotation/meshes/{mesh_name}"
        # mesh_file_path = f"{asset_root}/{mesh_name}"
        print(mesh_file_path)
        marker.mesh_resource = mesh_file_path
        marker.ns = mesh_file_path.split('/')[-1]
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        factor = self.mesh_size
        marker.scale.x = 1*factor
        marker.scale.y = 1*factor
        marker.scale.z = 1*factor
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1.0
        return marker

    # make the mesh controllable
    def make_marker_control(self, markers):
        control = InteractiveMarkerControl()
        control.always_visible = True
        for marker in markers:
            control.markers.append(marker)
        return control
        
    def create_6D_interactive_marker(self, obj_content_path, position=Point(0, 0, 0),):
        # add the menu to the maker
        self.menu_handler = MenuHandler()
        self.menu_handler.insert("show_pose", callback=self.showposeCallback)
        menu_handler = self.menu_handler.insert("record_pose", callback=self.emptyCallback)
        
        hammer_grip_menu_handle = self.menu_handler.insert( "hammer_grip", parent=menu_handler, callback=self.emptyCallback )
        self.menu_handler.insert( "head1", parent=hammer_grip_menu_handle, callback=self.processFeedback )
        self.menu_handler.insert( "grip", parent=hammer_grip_menu_handle, callback=self.processFeedback )
        
        screw_driver_menu_handle = self.menu_handler.insert( "screw_driver", parent=menu_handler, callback=self.emptyCallback )
        self.menu_handler.insert("head1", parent=screw_driver_menu_handle, callback=self.processFeedback)
        
        wrench_menu_handle = self.menu_handler.insert( "wrench", parent=menu_handler, callback=self.emptyCallback)
        self.menu_handler.insert("head1", parent=wrench_menu_handle, callback=self.processFeedback)
        self.menu_handler.insert("head2", parent=wrench_menu_handle, callback=self.processFeedback)
        

        # create the interactive marker
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.frame_id
        int_marker.pose.position = position
        int_marker.pose.orientation.w = 1
        int_marker.pose.orientation.x = 0
        int_marker.pose.orientation.y = 0
        int_marker.pose.orientation.z = 0
        int_marker.scale = self.marker_coord_size
        
        int_marker.name = obj_content_path
        int_marker.description = f"6-DOF interactive maker of {obj_content_path}"

        # create the mesh for marker
        mesh_marker = self.makeMesh(obj_content_path)
        # make the control for marker
        control = self.make_marker_control([mesh_marker])
        # add the control for interactive marker
        int_marker.controls.append(control)
        int_marker.controls[0].interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
        int_marker = self._add_6d_control(int_marker)
        # add the interactive marker to the server and don't call any callbacks which is empty
        self.server.insert(int_marker, self.emptyCallback)
        self.menu_handler.apply(self.server, int_marker.name)
        self.server.applyChanges()


    def _add_6d_control(self, int_marker):
        # mode = InteractiveMarkerControl.FIXED
        # # mode = InteractiveMarkerControl.ROTATE_AXIS
        
        # insert a rotation control along the X axis
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        normalizeQuaternion(control.orientation)
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        # insert a translation control along the Y axis
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        normalizeQuaternion(control.orientation)
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        # insert a rotation control along the Z axis
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        normalizeQuaternion(control.orientation)
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        # insert a translation control along the Z axis
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        normalizeQuaternion(control.orientation)
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        # insert a rotation control along the Y axis
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        normalizeQuaternion(control.orientation)
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        # insert a translation control along the Y axis
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        normalizeQuaternion(control.orientation)
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)
        return int_marker

# def get_todo_name(cat_name=None, obj_name=None):
#
#     if cat_name is not None and obj_name is not None:
#         return os.path.join(cat_name, f"{obj_name}.stl")
#
#     cats = os.listdir(os.path.join(data_root, new_dataset_name))
#     cat_list = [f for f in cats if os.path.isdir(os.path.join(data_root, new_dataset_name, f))]
#     for cat in cat_list:
#         cat_path = os.path.join(data_root, new_dataset_name, cat)
#         stl_list = [f for f in os.listdir(cat_path) if f.endswith(".stl")]
#         stl_list.sort()
#         for stl in stl_list:
#             stl_asset_path = os.path.join(cat, stl)
#             name = stl.split('.')[0]
#             norm_pose_path = os.path.join(cat_path, f"{name}_pose.txt")
#             if os.path.isfile(norm_pose_path):
#                 continue
#             else:
#                 return stl_asset_path, cat, name
#     return None, None

data_root = "/homeL/1wang/workspace/anno_ee_ws/src/pose_annotation/meshes/"
new_dataset_name = "Dataset3DModel_v3.0"
old_dataset_name = "Dataset3DModel_v3.0_norm_location_scale"
ee_pose_name = {
    "hammer_grip": ['head', 'handle'],
    "screwdriver": ['head'],
    "wrench": ['head1', 'head2']
}
def swicth_name(f1,f2):
    f_temp = f1+'.tmp'
    os.rename(f1, f_temp)
    os.rename(f2, f1)
    os.rename(f_temp, f2)
if __name__=="__main__":
   
    cat_name = "screwdriver"
    id = 16
    obj_name = f"screwdriver_{id:02d}"
    pose_head = os.path.join(data_root, new_dataset_name, cat_name, f"{obj_name}_head1_pose.txt")
    pose_grip = os.path.join(data_root, new_dataset_name, cat_name, f"{obj_name}_grip_pose.txt")
    # swicth_name(pose_head, pose_grip)
    
    # stl_asset_path = get_todo_name()
    stl_asset_path = os.path.join(new_dataset_name, cat_name, f"{obj_name}.stl")
    
    files = os.listdir(os.path.join(data_root, new_dataset_name, cat_name))
    stl_files = [f for f in files if f.endswith(".stl")]
    pose_files = [f for f in files if f.endswith("_pose.txt") and f.startswith(obj_name)]
    
    # norm_pose_path = os.path.join(data_root, dataset_name, cat_name, f"{obj_name}_norm_pose.txt")
    norm_pose_path = None
    
    rospy.init_node("marker_controls")
    # cat_list = os.listdir(asset_root)
    # cat_path = os.path.join(asset_root, cat_list[0])
    # obj_list = os.listdir(cat_path)
    # id = "01"
    # obj_content_path = os.path.join(cat_list[0], f"hammer_{id}.stl")
    imarker = Interactive6DMeshMarker(
        frame_id="map",
    )
    imarker.create_6D_interactive_marker(
        position=Point(0, 0, 0),
        obj_content_path=stl_asset_path)
    rospy.spin()

