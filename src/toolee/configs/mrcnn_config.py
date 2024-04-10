import argparse

mrcnn_seg_id_map = {
	"hammer_grip_head1": 0,
	"hammer_grip_grip": 1,
	"screwdriver_head1": 2,
	"wrench_head1": 3,
	"wrench_head2": 4,
}

def get_seg_id(name):
	return mrcnn_seg_id_map[name]

def get_all_ee_seg_names():
	return list(mrcnn_seg_id_map.keys())



def get_seg_name(id):
	for k, v in mrcnn_seg_id_map.items():
		if v == id:
			return k
	return None


def get_config():
	parser = argparse.ArgumentParser()
	
	""" dataset """
	parser.add_argument('--data_root', default='/dataSSD/yunlong/dataspace/DatasetToolEE', type=str)
	parser.add_argument('--data_dir', default='/dataSSD/yunlong/dataspace/DatasetToolEE_mrcnn', type=str)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--eval_batch_size', type=int, default=32)
	
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--num_workers', type=int, default=16)
	
	""" training """
	parser.add_argument('--lr', type=float, default=0.00025)
	parser.add_argument('--max_iter', type=int, default=40000)
	parser.add_argument('--num_classes', type=int, default=5)

	
	""" evaluation """
	parser.add_argument('--model_path', type=str, default='/dataSSD/yunlong/dataspace/mrcnn_result/output/model_final.pth')
	parser.add_argument('--eval_freq', type=int, default=0)
	
	""" testing """
	parser.add_argument('--roi_threshold', type=float, default=0.7)
	
	cfg = parser.parse_args()
	
	for k, v in cfg.__dict__.items():
		print(f'{k}: {v}')
	
	return cfg

