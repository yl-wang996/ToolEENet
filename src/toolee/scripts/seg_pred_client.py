import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
host_name = "tams110"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["ROS_MASTER_URI"] = f"http://{host_name}:11311"
# os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libtiff.so.5"
# os.environ["LD_LIBRARY_PATH"] = "/homeL/1wang/workspace/toolee_ws/devel/lib:/opt/ros/noetic/lib/x86_64-linux-gnu:/opt/ros/noetic/lib"

from utils.file_utils import MetaUtils
import time
import rospy
import numpy as np
from tool_ee.srv import SegPred, SegPredRequest
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()

def call_seg_pred_service(rgb_img: np.ndarray, vis=True):
	
	try:
		rospy.wait_for_service('seg_pred', timeout=10)
		rospy.loginfo('seg_pred service is available')
	except rospy.ROSException:
		rospy.logerr('seg_pred service is not available')
		return None
	try:
		seg_pred_service = rospy.ServiceProxy('seg_pred', SegPred)
		
		request_msg = SegPredRequest(rgb=bridge.cv2_to_imgmsg(rgb_img, encoding='passthrough'), vis=vis)
		response_meg = seg_pred_service(request_msg)
		seg_msg_list = response_meg.seg_list
		seg_name_list = response_meg.seg_name_list
		seg_masks = []
		for idx, _ in enumerate(seg_name_list):
			img_msg = seg_msg_list[idx]
			seg_img = bridge.imgmsg_to_cv2(img_msg)
			seg_mask = np.zeros_like(seg_img, dtype=bool)
			seg_mask[seg_img == 255] = True
			seg_mask[seg_img == 0] = False
			seg_masks.append(seg_mask)
		return seg_masks, seg_name_list
		
	except rospy.ServiceException as e:
		print("Service call failed: %s" % e)


if __name__ == '__main__':
	'''
	if error happened with cv_bridge when using conda env, try the following command before run the python script:
		export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/noetic/lib
	'''

	rospy.init_node('seg_prediction_node_test', log_level=rospy.DEBUG)
	# dataset_root = "/dataSSD/yunlong/dataspace/DatasetToolEE"
	# cat_name = "hammer_grip"  # "hammer_grip", "screwdriver", "wrench"
	# obj_name = "hammer_02"
	# ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	# # ids = [0, ]
	# for id in ids:
	# 	meta_name = f"meta_{cat_name}_{obj_name}_{id:04d}.yaml"
	# 	meta_path = os.path.join(dataset_root, cat_name, obj_name, meta_name)
	# 	meta_util = MetaUtils(data_root=dataset_root, meta_name=meta_path)
	# 	rgb_img = meta_util.get_image()
	# 	seg_img_gt = meta_util.get_affordance_seg()
	# 	call_seg_pred_service(rgb_img, vis=True)
	#
	# 	time.sleep(2)
	# print('done!')
	img = cv2.imread("/homeL/1wang/workspace/toolee_ws/src/toolee/rgb.png")
	rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	call_seg_pred_service(rgb_img, vis=True)
	