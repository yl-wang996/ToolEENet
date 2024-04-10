import tf
import rospy
import numpy as np

def load_pose():
    id = "05"
    pth = f'/homeL/1wang/workspace/toolee_ws/src/pose_annotation/meshes/Dataset3DModel_v2.0/Hammer_Grip/hammer_{id}_head_pose.txt'
    pose = np.loadtxt(pth, delimiter=',')
    return pose
def publish_tf():
    pose = load_pose()
    trans, quat = tf.transformations.translation_from_matrix(pose), tf.transformations.quaternion_from_matrix(pose)
    while not rospy.is_shutdown():
        br = tf.TransformBroadcaster()
        br.sendTransform(trans,
                         quat,
                         rospy.Time.now(),
                         "camera_link",
                         "base_link")
        rospy.sleep(0.1)

if __name__ == '__main__':
    rospy.init_node('tf_broadcaster')
    publish_tf()