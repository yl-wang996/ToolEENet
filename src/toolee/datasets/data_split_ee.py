import os
import random

from tqdm import tqdm
import numpy as np

from utils.file_utils import MetaUtils

if __name__ == '__main__':
	random.seed(0)
	train_ratio = 0.8
	data_path = '/dataSSD/yunlong/dataspace/DatasetToolEE'
	exclude_objs = ['hammer_10', 'hammer_11']  # some problems with those 3D models
	cats = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
	all_training_examples = []
	novel_examples = []
	cats.sort()
	for cat in cats:
		objs = [f for f in os.listdir(os.path.join(data_path, cat)) if os.path.isdir(os.path.join(data_path, cat, f))]
		objs.sort()
		for obj in objs[:-1]:
			if obj in exclude_objs:
				continue
			meta_file_names = [f for f in os.listdir(os.path.join(data_path, cat, obj)) if 'meta' in f]
			for meta_file_name in tqdm(meta_file_names):
				meta_file_path = os.path.join(cat, obj, meta_file_name)
				meta_util = MetaUtils(data_path, meta_file_path)
				ee_names = meta_util.get_ee_names()
				for ee_name in ee_names:
					all_training_examples.append([meta_file_path, ee_name])
					
		novel_obj = objs[-1]
		meta_file_names = [f for f in os.listdir(os.path.join(data_path, cat, novel_obj)) if 'meta' in f]
		for meta_file_name in tqdm(meta_file_names):
			meta_file_path = os.path.join(cat, novel_obj, meta_file_name)
			meta_util = MetaUtils(data_path, meta_file_path)
			for ee_name in meta_util.get_ee_names():
				novel_examples.append([meta_file_path, ee_name])
	
	novel_examples = np.asarray(novel_examples)
	all_training_examples = np.asarray(all_training_examples)
	print('save to file')
	np.savetxt(os.path.join(data_path, 'novel_examples_ee.txt'), novel_examples, fmt='%s', delimiter=',')
	np.savetxt(os.path.join(data_path, 'all_training_examples_ee.txt'), all_training_examples, fmt='%s', delimiter=',')
	
	np.random.shuffle(all_training_examples)
	train_num = int(len(all_training_examples) * train_ratio)
	train_examples = all_training_examples[:train_num,:]
	val_examples = all_training_examples[train_num:,:]
	print('save to file')
	np.savetxt(os.path.join(data_path, 'train_examples_ee.txt'), train_examples, fmt='%s', delimiter=',')
	np.savetxt(os.path.join(data_path, 'val_examples_ee.txt'), val_examples, fmt='%s', delimiter=',')





