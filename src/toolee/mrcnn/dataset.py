import cv2
from detectron2.utils.logger import setup_logger

setup_logger()

import numpy as np
import os
from tqdm import tqdm
import random
import pickle

from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools.mask import encode as mask_encode
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode

from utils.file_utils import MetaUtils
from configs.mrcnn_config import get_seg_id, get_all_ee_seg_names, get_config

def get_dataset_list(data_root, meta_file_name='_examples_ee_visible.txt'):
	examples = np.loadtxt(os.path.join(data_root, meta_file_name), dtype=str, delimiter=',')
	return examples

def dataset_visualization(meta_data, dataset, cfg):
	tmp_folder = os.path.join(os.path.dirname(cfg.data_root), 'mrcnn_result', "tmp")
	os.makedirs(tmp_folder, exist_ok=True)
	assert dataset in meta_data.name, f"dataset {dataset} not found"
	dataset_dicts = extract_datasets_dicts(dataset=dataset, cfg=cfg)
	os.makedirs(tmp_folder, exist_ok=True)
	for d in random.sample(dataset_dicts, 3):
		image_path = d["file_name"]
		file_name = os.path.basename(image_path)
		img = cv2.imread(image_path)
		visualizer = Visualizer(img[:, :, ::-1], metadata=meta_data, scale=0.5)
		out = visualizer.draw_dataset_dict(d)
		cv2.imwrite(filename=os.path.join(tmp_folder, f"vis_{file_name}.jpg"), img=out.get_image()[:, :, ::-1])

def extract_datasets_dicts(dataset='train', cfg=None):
	assert dataset in ['train', 'val', 'novel'], f"dataset {dataset} not found"
	dataset_file_name = f"{dataset}_examples_ee_visible.txt"
	data_root = cfg.data_root
	examples = get_dataset_list(data_root, dataset_file_name)
	
	dataset_list = []
	for meta_file_name, _ in tqdm(examples, desc=f"loading {dataset} dataset"):
		meta_util = MetaUtils(data_root, meta_file_name)
		cat, _, _ = meta_util.get_cat_obj_id()
		height, width = meta_util.get_cam_hw()
		image_path = os.path.join(data_root, meta_util.image_path)
		record = {}
		
		record["file_name"] = image_path
		record["image_id"] = meta_file_name  # str
		record["height"] = height
		record["width"] = width
		
		seg_array = meta_util.get_affordance_seg()
		ee_points_dict = meta_util.get_ee_points()
		if len(ee_points_dict) == 0:
			print(f"no ee points found in {meta_file_name}")
		
		ids = np.unique(seg_array)
		annotations = []
		
		for ee_name, ee_point in ee_points_dict.items():
			ee_id = get_seg_id(f"{cat}_{ee_name}")
			seg_id = ee_id + 3
			if seg_id not in ids:
				continue
			anno = {}
			bit_seg_array = seg_array == seg_id
			ys, xs = np.where(bit_seg_array == True)
			x_min, x_max = np.min(xs), np.max(xs)
			y_min, y_max = np.min(ys), np.max(ys)
			# bbox = [np.max([x_min, 0]), np.max([y_min, 0]), np.min([x_max, height]), np.min([y_max, width])]
			bbox = [x_min, y_min, x_max, y_max]
			anno["bbox"] = bbox
			anno["bbox_mode"] = BoxMode.XYXY_ABS
			anno["category_id"] = ee_id
			mask = mask_encode(np.asarray(bit_seg_array.astype(np.uint8),
			                              order="F"))  # set cfg.INPUT.MASK_FORMAT to 'bitmask' if using the default data loader with such format.
			anno['segmentation'] = mask
			# [x, y, v],  v=1 means visible 2 means invisible 0 means not labeled
			# anno['keypoints'] = [int(ee_point[0]), int(ee_point[1]), 1]
			anno['iscrowd'] = 0
			annotations.append(anno)
			if len(ee_points_dict) != 0 and len(annotations)==0:
				print(f"no annotations found in {meta_file_name}")
		if len(ee_points_dict) != 0 and len(annotations) == 0:
			print(f"no annotations found in {meta_file_name}")
		record['annotations'] = annotations
		dataset_list.append(record)
	return dataset_list

def save_dataset_dicts(cfg, data_dir):
	for dataset in ["train", 'val', 'novel']:
		dataset_path = os.path.join(data_dir, f"{dataset}_dataset_dicts.pkl")
		if not os.path.exists(dataset_path):
			dataset_list = extract_datasets_dicts(dataset, cfg)
			with open(dataset_path, 'wb') as f:
				pickle.dump(dataset_list, f)
		else:
			print(f"dataset {dataset} already exists, skip saving")

def load_dataset_dicts(dataset, data_dir):
	dataset_path = os.path.join(data_dir, f"{dataset}_dataset_dicts.pkl")
	with open(dataset_path, 'rb') as f:
		dataset_list = pickle.load(f)
	return dataset_list

def get_meta_data(dataset='val'):
	meta_data = MetadataCatalog.get("ToolEE/" + dataset)
	return meta_data

def register_datasets(cfg):
	for dataset in ["train", 'val', 'novel']:
		DatasetCatalog.register("ToolEE/" + dataset, lambda d=dataset: load_dataset_dicts(dataset, cfg.data_dir))
		ee_seg_names = get_all_ee_seg_names()
		meta_data = MetadataCatalog.get("ToolEE/" + dataset).set(
			thing_classes=ee_seg_names,
		)
		vis = False
		if vis:
			dataset_visualization(meta_data=meta_data, dataset=dataset, cfg=cfg)

if __name__ == '__main__':
	cfg = get_config()
	data_dir = cfg.data_dir
	os.makedirs(data_dir, exist_ok=True)
	save_dataset_dicts(cfg,data_dir)
	register_datasets(cfg)
	train_dict = load_dataset_dicts("train", data_dir)
	val_dict = load_dataset_dicts("val", data_dir)
	novel_dict = load_dataset_dicts("novel", data_dir)
	
	print(f"train dataset has {len(train_dict)} images")
	print(f"val dataset has {len(val_dict)} images")
	print(f"novel dataset has {len(novel_dict)} images")