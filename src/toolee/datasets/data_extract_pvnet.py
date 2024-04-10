import os.path
import open3d as o3d
from utils.file_utils import MetaUtils
import numpy as np
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import open3d as o3d

def load_mesh(cat_name, obj_name, scale):
	
	asset_root = '/dataSSD/yunlong/dataspace/Dataset3DModel/'
	mesh_path = os.path.join(asset_root, cat_name, f'{obj_name}.stl')
	if not os.path.exists(mesh_path):
		raise ValueError(f"the pcd file {mesh_path} does not exist!")
	# load mesh file
	mesh = o3d.io.read_triangle_mesh(mesh_path)
	mesh.scale(scale, center=np.array([0,0,0]))
	return mesh

def convert_dataset(example):
	meta_name, ee_name = example
	data_root_target = '/dataSSD/yunlong/dataspace/DatasetToolEE_pvnet'
	data_root = '/dataSSD/yunlong/dataspace/DatasetToolEE'
	meta_util = MetaUtils(data_root, meta_name)
	cat_name, obj_name, idx = meta_util.get_cat_obj_id()
	obj_dir = os.path.join(data_root_target, cat_name, obj_name)
	os.makedirs(obj_dir, exist_ok=True)
	
	# save scale
	scale = meta_util.get_obj_scale()
	scale_folder = os.path.join(obj_dir, f'{scale:.4f}')
	os.makedirs(scale_folder, exist_ok=True)
	
	# save camera view matrix
	camera_view = meta_util.get_cam_view_matrix()
	camera_view_folder = os.path.join(scale_folder, 'camera_view_matrix')
	os.makedirs(camera_view_folder, exist_ok=True)
	np.save(os.path.join(camera_view_folder, f'view{idx}.npy'), camera_view)
	
	# save rgb image
	rgb_image = meta_util.get_image()
	rgb_image_folder = os.path.join(scale_folder, 'rgb')
	os.makedirs(rgb_image_folder, exist_ok=True)
	rgb_image_path = os.path.join(rgb_image_folder, f'{idx}.jpg')
	if os.path.exists(rgb_image_path):
		return
	rgb_image = Image.fromarray(rgb_image)
	rgb_image.save(rgb_image_path)
	

	
	# save obj pose
	obj_pose = meta_util.get_obj_pose()
	obj_pose_folder = os.path.join(scale_folder, 'pose')
	os.makedirs(obj_pose_folder, exist_ok=True)
	np.save(os.path.join(obj_pose_folder, f'pose{idx}.npy'), obj_pose)
	ee_pose_dict = meta_util.get_ee_poses()
	
	# save ee pose
	for k, v in ee_pose_dict.items():
		ee_pose_folder = os.path.join(scale_folder, f"ee_pose_{k}")
		os.makedirs(ee_pose_folder, exist_ok=True)
		os.makedirs(ee_pose_folder, exist_ok=True)
		np.save(os.path.join(ee_pose_folder, f'pose{idx}.npy'), v)
	
	# save mask image
	mask_image = meta_util.get_seg()
	ids = np.unique(mask_image)
	obj_id = meta_util.get_obj_seg_id()
	mask_image = mask_image == obj_id
	im = Image.fromarray(np.uint8((mask_image) * 255))
	mask_image_folder = os.path.join(scale_folder, 'mask')
	os.makedirs(mask_image_folder, exist_ok=True)
	mask_image_path = os.path.join(mask_image_folder, f'{idx}.png')
	im.save(mask_image_path)
	
	# convert mesh
	mesh = load_mesh(cat_name, obj_name, scale=scale)
	mesh.compute_vertex_normals()
	mesh.compute_triangle_normals()
	mesh_path = os.path.join(scale_folder, 'model.stl')
	o3d.io.write_triangle_mesh(mesh_path, mesh)
	
	# save projection matrix
	proj_matirx = meta_util.get_cam_proj_matrix()
	proj_matirx = proj_matirx[:3, :3]
	u0 = int(1080/2)
	v0 = int(1920/2)
	proj_matirx[0, 2] = u0
	proj_matirx[1, 2] = v0
	proj_matrix_path = os.path.join(scale_folder, 'camera.txt')
	np.savetxt(proj_matrix_path, proj_matirx)
	


if __name__ == '__main__':
	
	data_root = '/dataSSD/yunlong/dataspace/DatasetToolEE'

	all_train_examples = np.loadtxt(os.path.join(data_root, 'all_training_examples_ee_visible.txt'), dtype=str,
	                                delimiter=',').tolist()
	novel_examples = np.loadtxt(os.path.join(data_root, 'novel_examples_ee_visible.txt'), dtype=str, delimiter=',').tolist()

	with Pool(processes=50) as pool:
		for _ in tqdm(pool.imap_unordered(convert_dataset, all_train_examples), total=len(all_train_examples)):
			pass
		
	with Pool(processes=50) as pool:
		for _ in tqdm(pool.imap_unordered(convert_dataset, novel_examples), total=len(novel_examples)):
			pass
		




		

	
	