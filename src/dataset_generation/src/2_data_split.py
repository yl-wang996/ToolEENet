import os
import random

if __name__ == '__main__':
	random.seed(0)
	train_ratio = 0.8
	data_path = '/dataSSD/1wang/dataspace/DatasetToolEE'
	exclude_objs = ['hammer_10', 'hammer_11']  # some problems with those 3D models
	cats = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
	all_examples = []
	for cat in cats:
		objs = [f for f in os.listdir(os.path.join(data_path, cat)) if os.path.isdir(os.path.join(data_path, cat, f))]
		for obj in objs:
			if obj in exclude_objs:
				continue
			meta_file_names = [f for f in os.listdir(os.path.join(data_path, cat, obj)) if 'meta' in f]
			all_examples += [os.path.join(cat, obj, f) for f in meta_file_names]

	all_examples.sort()
	with open(os.path.join(data_path, 'all_examples.txt'), 'w') as f:
		for example_id in all_examples:
			f.write(example_id + '\n')

	random.shuffle(all_examples)
	train_num = int(len(all_examples) * train_ratio)
	train_examples = all_examples[:train_num]
	val_examples = all_examples[train_num:]

	with open(os.path.join(data_path, 'train_examples.txt'), 'w') as f:
		for example_id in train_examples:
			f.write(example_id + '\n')

	with open(os.path.join(data_path, 'val_examples.txt'), 'w') as f:
		for example_id in val_examples:
			f.write(example_id + '\n')




