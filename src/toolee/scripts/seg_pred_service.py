import os
import sys

import numpy as np

host_name = "tams110"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["ROS_MASTER_URI"] = f"http://{host_name}:11311"
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import rospy
from tool_ee.srv import SegPred, SegPredResponse

from configs.mrcnn_config import get_config
from mrcnn.runner import get_predictor
from cv_bridge import CvBridge,CvBridgeError
import random
import cv2
from detectron2.utils.visualizer import Visualizer
from mrcnn.dataset import get_meta_data
from detectron2.utils.visualizer import ColorMode
from sensor_msgs.msg import Image

seg_id_map_valid = {
	0: "hammer_grip_head1",
	1: "hammer_grip_grip",
	2: "screwdriver_head1",
	3: "wrench_head1",
}

# the inference class
class AffSegPrediction(object):
	def __init__(self):
		self.val_metadata = get_meta_data(dataset='val')
		self.train_metadata = get_meta_data(dataset='train')
	
	def init_model(self, ckpt_path=None):
		if ckpt_path is None:
			ckpt_path = "/dataSSD/yunlong/dataspace/mrcnn_result/output/model_0024999.pth"
		self.predictor = get_predictor(ckpt_path=ckpt_path,roi_threshold=0.7)
		rospy.loginfo('model initialized')
	
	def predict(self, img, visualize=True, region_filter=True):
		''' predict the affordance segmentation '''
		"""
			aff_seg_pred: [bs, H, W]
			classed_ids: [bs, num_instances]
			result_draw: [H, W, 3]
		"""

		
		pred_result = self.predictor(img)
		instances = pred_result['instances'].to('cpu')
		if region_filter:
			fesible_ids = []
			h = img.shape[0]
			w = img.shape[1]
			fesible_region = np.zeros((h, w), dtype=bool)
			fesible_region[int(h / 3):int(h / 3 * 2), int(w / 3):int(w / 3 * 2)] = True
			for i in range(len(instances)):
				bbox = instances.pred_boxes.tensor[i].numpy()
				bbox_center = [int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3]) / 2)]
				if fesible_region[bbox_center[1], bbox_center[0]]:
					fesible_ids.append(i)
			instances = instances[fesible_ids]
			pred_result['instances'] = instances
		classed_ids = instances.pred_classes.numpy()
		aff_seg_pred = instances.pred_masks.numpy()
		result_draw = None
		if visualize:

			result_draw = self.draw_result(img, instances)

		return pred_result, classed_ids, aff_seg_pred, result_draw
	
	def draw_result(self, img, instances, save_path=None):
		v = Visualizer(
			img[:, :, ::-1],
			metadata=self.val_metadata,
			scale=1.0,
			instance_mode=ColorMode.IMAGE
			# remove the colors of unsegmented pixels. This option is only available for segmentation models
		)
		out = v.draw_instance_predictions(instances.to("cpu"))
		img_result = out.get_image()[:, :, ::-1]
		if save_path is not None:
			cv2.imwrite(filename=save_path, img=img_result)
		return img_result
	
	def warm_up(self):
		''' warm up the model '''
		rospy.loginfo('warming up the model')
		img = np.zeros((1920, 1080, 3))
		self.predictor(img)

# the ros node wrapper for the inference class
class AffSegPredictionNode():
	def __init__(self):
		self.bridge = CvBridge()
		self.aff_seg_pred = AffSegPrediction()
		self.aff_seg_pred.init_model()
		self.vis_pub = rospy.Publisher('seg_vis', Image, queue_size=1, latch=True)
	
	def warmup(self):
		''' warm up the model '''
		self.aff_seg_pred.warm_up()
		rospy.loginfo('affordance segmentation model warmed up')
	
	def start_service(self,service_name='seg_pred'):
		''' spin up the node '''
		rospy.Service(service_name, SegPred, self.prediction_service)
	
	def pub_result(self, img):
		''' publish the visualization result '''
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		vis_img_msg = self.bridge.cv2_to_imgmsg(img, 'bgr8')
		self.vis_pub.publish(vis_img_msg)
	
	def prediction_service(self, msg):
		rospy.logdebug('received prediction request')
		vis = msg.vis
		try:
			rgb_img = self.bridge.imgmsg_to_cv2(msg.rgb, 'rgb8')
		except CvBridgeError as e:
			rospy.logerr(f'failed to convert the image message: {e}')
			return SegPredResponse()
		pred_result, classed_ids, aff_seg_pred, result_draw = self.aff_seg_pred.predict(rgb_img, visualize=vis)
		if vis:
			self.pub_result(result_draw)
		seg_list = []
		seg_name_list = []
		for idx, classed_id in enumerate(classed_ids):
			if classed_id not in seg_id_map_valid:
				continue
			# wrap the binary segmentation to the image message
			# aff_seg_pred: [bs, H, W] each value is (Ture/False)
			seg_img = np.zeros_like(aff_seg_pred[idx])
			seg_img[aff_seg_pred[idx] == True] = 1
			seg_img = seg_img.astype(np.uint8)
			# seg_img = np.repeat(seg_img[:, :, np.newaxis], 3, axis=2)
			# cv_rgb_image = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
			seg_msg = self.bridge.cv2_to_imgmsg(seg_img, encoding='mono8')
			seg_list.append(seg_msg)
			seg_name_list.append(seg_id_map_valid[classed_id])
		rospy.logdebug(f'prediction: {len(seg_name_list)} segments are predicted')
		return SegPredResponse(seg_list=seg_list, seg_name_list=seg_name_list)

# todo: finish the seg prediction service
if __name__ == '__main__':
	# init the node
	rospy.init_node('seg_pred_node', log_level=rospy.DEBUG)
	seg_pred_node = AffSegPredictionNode()
	seg_pred_node.warmup()
	seg_pred_node.start_service('seg_pred')
	try:
		rospy.wait_for_service('seg_pred', timeout=10)
		rospy.loginfo('seg_pred service is available')
	except rospy.ROSException:
		rospy.logerr('seg_pred service not available')
		rospy.signal_shutdown('seg_pred service not available')
	rospy.spin()
