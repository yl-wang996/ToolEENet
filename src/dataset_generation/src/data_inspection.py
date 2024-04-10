import numpy as np
import point_cloud_utils as pcu
import open3d as o3d
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def is_head_exist(cat_name, obj_name, ee_name, model3d_root='/home/weiweidu/Downloads/Model3D'):
	cat_path = os.path.join(model3d_root, cat_name)
	pose_file_name = f"{obj_name}_{ee_name}_pose.txt"
	ee_path = os.path.join(cat_path, pose_file_name)
	if os.path.exists(ee_path):
		return True
	else:
		return False

def load_pcd_from_stl(stl_path, target_num_pts=10000):
	t0 = time.time()
	# v is a [n, 3] shaped NumPy array of vertices
	# f is a [m, 3] shaped integer NumPy array of indices into v
	# n is a [n, 3] shaped NumPy array of vertex normals
	v, f, n = pcu.load_mesh_vfn(stl_path)

	### Generating points according to a blue noise distribution with a target number of points
	### NOTE: The number of returned points may not be exactly 1000
	# Generate barycentric coordinates of random samples
	fid, bc = pcu.sample_mesh_poisson_disk(v, f, num_samples=target_num_pts*2)

	# Interpolate the vertex positions and normals using the returned barycentric coordinates
	# to get sample positions and normals
	rand_positions = pcu.interpolate_barycentric_coords(f, fid, bc, v)
	print(f"Time to load pcd from stl: {time.time() - t0:.2f}s")
	return rand_positions

def extract_ee_pcd_by_distance(obj_pcd, ee_trans, distance_threshold=0.1):
	t0 = time.time()
	distances = np.linalg.norm(obj_pcd - ee_trans, axis=1)
	ee_pcd = obj_pcd[distances < distance_threshold]
	return ee_pcd

def calc_chamfer_distance(pcd1, pcd2):
	distance = pcu.chamfer_distance(pcd1, pcd2)
	return distance

def calc_hausdorff_distance(pcd1, pcd2):
	distance = pcu.hausdorff_distance(pcd1, pcd2)
	return distance

def calc_earth_movers_distance(pcd1, pcd2):
	distance = pcu.earth_movers_distance(pcd1, pcd2)
	return distance

def rotate_pcd(pcd, R):
	'''
		R (numpy.ndarray[numpy.float64[3, 3]]): The rotation matrix
	'''
	o3d_pcd = o3d.geometry.PointCloud()
	o3d_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd))
	o3d_pcd.rotate(R, center=(0,0,0))
	rotated_pcd = np.asarray(o3d_pcd.points)
	return rotated_pcd

def normalize_ee_pcd(pcd, RT):
	iRT = np.linalg.inv(RT)
	trans = iRT[:3, 3]
	R = iRT[:3, :3]
	pcd = rotate_pcd(pcd, R)
	pcd = pcd + trans
	return pcd

def pcd_downsampling(pcd, num=500):
	'''
		Downsample the point cloud to a fixed number of points
	'''
	assert len(pcd) > num, f"the input pcd has less than {num} points"
	indices = np.random.choice(len(pcd), num, replace=False)
	downsampled_pcd = pcd[indices]
	return downsampled_pcd, indices


def draw_confuse_matrix(matrix, labels, save_path,title='Confusion matrix', is_show=False):
	labels = [label.split('_')[-1] for label in labels]
	df_cm = pd.DataFrame(matrix, index=labels, columns=labels)
	df_cm = df_cm.round(2)
	# plt.figure(figsize=(10,7))
	sns.set(font_scale=0.8)  # for label size
	sns.heatmap(df_cm, annot=True, annot_kws={"size": 6})  # font size
	plt.title(title)
	if is_show:
		plt.show()
	print(f"save confuse_matrix plot to {save_path}")
	plt.savefig(save_path)
	plt.close()

def draw_mean_var():
	fig, ax = plt.subplots()
	for folder_name in ['ee_pcds_vis', 'ee_pcds_norm_vis']:
		data = np.loadtxt(os.path.join(model3d_root, folder_name, 'overall_statistical_result.csv'),delimiter=',', dtype=str)
		means = []
		vars = []
		names = []
		for key, value in data:
			name = key.split('_')[0].upper()
			if name not in names:
				names.append(name)
			if 'mean' in key:
				means.append(float(value))
			if 'std' in key:
				vars.append(float(value))
		if 'norm' in folder_name:
			ax.bar(names, means, yerr=vars, align='center', alpha=0.5, ecolor='red', capsize=10)
		else:
			ax.bar(names, means, yerr=vars, align='center', alpha=0.5, ecolor='green', capsize=10)

	ax.set_xlabel('Distance Metrics')
	ax.set_ylabel('Distance Value')
	# draw the legend
	ax.legend(['object-centric', 'ee-centric'])
	ax.set_xticks(names)
	ax.set_title('Mean and Variance of Average Points Cloud Distance')
	ax.yaxis.grid(True)
	plt.savefig(os.path.join(model3d_root, 'overall_statistical_result.png'))
	plt.show()



def visualize_result(model3d_root, overwrite_vis=False):
	# visualize the statistical result and draw the confuse matrix
	for pcd_foler_name in ['ee_pcds', 'ee_pcds_norm']:
		cd_avg = []
		hd_avg = []
		emd_avg = []


		distance_folder_name = 'pcd_distance'
		vis_folder_name = pcd_foler_name + '_vis'
		vis_folder = os.path.join(model3d_root, vis_folder_name)
		if os.path.exists(vis_folder) and not overwrite_vis:
			print(f"{vis_folder_name} already existed, skip")
			continue
		os.makedirs(vis_folder, exist_ok=True)

		pcd_folder = os.path.join(model3d_root, pcd_foler_name)
		cat_names = os.listdir(pcd_folder)
		cat_names.sort()
		for cat_name in cat_names:
			ee_names = np.unique(
				[f.replace('.npy', '').split('_')[-1] for f in os.listdir(os.path.join(pcd_folder, cat_name)) if
				 f.endswith('.npy')])
			ee_names.sort()
			for ee_name in ee_names:
				if 'head2' in ee_name:
					continue
				distance_folder = os.path.join(pcd_folder, cat_name, distance_folder_name)
				cd_matrix = np.load(os.path.join(distance_folder, f"{ee_name}_cd_matrix.npy"))
				hd_matrix = np.load(os.path.join(distance_folder, f"{ee_name}_hd_matrix.npy"))
				emd_matrix = np.load(os.path.join(distance_folder, f"{ee_name}_emd_matrix.npy"))
				# draw the confuse matrix
				obj_names = np.unique(
					[f.replace(f'_{ee_name}.npy', '') for f in os.listdir(os.path.join(pcd_folder, cat_name)) if
					 f.endswith('.npy') and ee_name in f])
				obj_names.sort()
				draw_confuse_matrix(cd_matrix, labels=obj_names,
				                    save_path=os.path.join(vis_folder, f"{cat_name}_{ee_name}_cd_matrix.png"),
				                    title=f'Chamfer distance of {cat_name}_{ee_name}')
				draw_confuse_matrix(hd_matrix, labels=obj_names,
				                    save_path=os.path.join(vis_folder, f"{cat_name}_{ee_name}_hd_matrix.png"),
				                    title=f'Hausdorff distance of {cat_name}_{ee_name}')
				draw_confuse_matrix(emd_matrix, labels=obj_names,
				                    save_path=os.path.join(vis_folder, f"{cat_name}_{ee_name}_emd_matrix.png"),
				                    title=f'Earth Movers distance of {cat_name}_{ee_name}')
				# draw the statistical result
				cd = np.mean(cd_matrix, axis=1)
				cd_mean = np.mean(cd, axis=0)
				cd_std = np.std(cd, axis=0)
				hd = np.mean(hd_matrix, axis=1)
				hd_mean = np.mean(hd, axis=0)
				hd_std = np.std(hd, axis=0)
				emd = np.mean(emd_matrix, axis=1)
				emd_mean = np.mean(emd, axis=0)
				emd_std = np.std(emd, axis=0)
				cd_avg.append(cd_mean)
				hd_avg.append(hd_mean)
				emd_avg.append(emd_mean)
				# save the result to csv
				_result = [
					['cd_mean', str(cd_mean)],
					['cd_std', str(cd_std)],
					['hd_mean', str(hd_mean)],
					['hd_std', str(hd_std)],
					['emd_mean', str(emd_mean)],
					['emd_std', str(emd_std)],
				]
				np.savetxt(os.path.join(vis_folder, f"{cat_name}_{ee_name}_statistical_result.csv"), _result,
				           delimiter=',', fmt='%s')
				print(
					f"save the statistical result to {os.path.join(vis_folder, f'{cat_name}_{ee_name}_statistical_result.csv')}")
		cd_avg_mean = np.mean(cd_avg, axis=0)
		cd_avg_std = np.std(cd_avg, axis=0)
		hd_avg_mean = np.mean(hd_avg, axis=0)
		hd_avg_std = np.std(hd_avg, axis=0)
		emd_avg_mean = np.mean(emd_avg, axis=0)
		emd_avg_std = np.std(emd_avg, axis=0)
		# save the result to csv
		_result = [
			['cd_mean', str(cd_avg_mean)],
			['cd_std', str(cd_avg_std)],
			['hd_mean', str(hd_avg_mean)],
			['hd_std', str(hd_avg_std)],
			['emd_mean', str(emd_avg_mean)],
			['emd_std', str(emd_avg_std)],
		]
		np.savetxt(os.path.join(vis_folder, f"overall_statistical_result.csv"), _result,
		           delimiter=',', fmt='%s', )
		print(f"save the overall statistical result to {os.path.join(vis_folder, 'overall_statistical_result.csv')}")



def compute_distance(model3d_root, overwrite_distance=False):
	# start to calculate the distance
	for pcd_foler_name in ['ee_pcds', 'ee_pcds_norm']:
		distance_folder_name = 'pcd_distance'
		pcd_folder_path = os.path.join(model3d_root, pcd_foler_name)
		cat_names = os.listdir(pcd_folder_path)
		cat_names.sort()
		for cat_name in cat_names:
			distance_folder_path = os.path.join(pcd_folder_path, cat_name, distance_folder_name)
			if os.path.exists(distance_folder_path) and not overwrite_distance:
				print(f"{distance_folder_path} already existed, skip")
				continue
			os.makedirs(distance_folder_path,exist_ok=True)
			print(f"start to calculate the {distance_folder_path}...")
			ee_names = np.unique([f.replace('.npy','').split('_')[-1] for f in os.listdir(os.path.join(pcd_folder_path, cat_name)) if f.endswith('.npy')])
			ee_names.sort()
			for ee_name in ee_names:
				if 'head2' in ee_name:
					continue
				pcd_names = [f for f in os.listdir(os.path.join(pcd_folder_path, cat_name)) if f.endswith('.npy') and ee_name in f]
				pcd_names.sort()
				pcd_names = pcd_names
				cd_matrix = np.zeros((len(pcd_names), len(pcd_names)))
				emd_matrix = np.zeros((len(pcd_names), len(pcd_names)))
				hd_matrix = np.zeros((len(pcd_names), len(pcd_names)))
				print(f'--'*10+pcd_foler_name+'--'*10)
				for idx_1, pcd_name_1 in enumerate(pcd_names):
					for idx_2, pcd_name_2 in enumerate(pcd_names):

						pcd_1 = np.load(os.path.join(pcd_folder_path, cat_name, pcd_name_1))
						pcd_2 = np.load(os.path.join(pcd_folder_path, cat_name, pcd_name_2))
						cd_matrix[idx_1,idx_2] = calc_chamfer_distance(pcd_1, pcd_2)
						hd_matrix[idx_1,idx_2] = calc_hausdorff_distance(pcd_1, pcd_2)
						emd_matrix[idx_1,idx_2], _ = calc_earth_movers_distance(pcd_1, pcd_2)
						print(f"{pcd_name_1} vs {pcd_name_2} cd: {cd_matrix[idx_1,idx_2]:.3f}, hd: {hd_matrix[idx_1,idx_2]:.3f}, emd: {emd_matrix[idx_1,idx_2]:.3f}")
				distance_folder = os.path.join(pcd_folder_path, cat_name, distance_folder_name)
				if not os.path.exists(distance_folder):
					os.makedirs(distance_folder, exist_ok=True)

				is_save = True
				if is_save:
					with open(os.path.join(distance_folder, f"{ee_name}_cd_matrix.npy"), 'wb') as f:
						np.save(f, cd_matrix)
					with open(os.path.join(distance_folder, f"{ee_name}_hd_matrix.npy"), 'wb') as f:
						np.save(f, hd_matrix)
					with open(os.path.join(distance_folder, f"{ee_name}_emd_matrix.npy"), 'wb') as f:
						np.save(f, emd_matrix)

def extract_ee_pcd(model3d_root, distance_threshold, overwrite_pcd=False, random_scale_factor=0.0):

	# start to generate the pcds
	for pcd_foler_name in ['ee_pcds', 'ee_pcds_norm']:
		pcd_tmp_folder = os.path.join(model3d_root, pcd_foler_name)
		if os.path.exists(pcd_tmp_folder) and not overwrite_pcd:
			print(f"{pcd_foler_name} already exist in {model3d_root}, skip")
			continue

		os.makedirs(pcd_tmp_folder, exist_ok=True)
		cat_names = [f for f in os.listdir(model3d_root) if os.path.isdir(os.path.join(model3d_root, f))]
		cat_names.sort()
		obj_names = []
		ee_names = []
		for cat_name in cat_names:
			_ee_name = []
			_obj_names = [f.replace('.stl', '') for f in os.listdir(os.path.join(model3d_root, cat_name)) if
			              f.endswith('.stl')]
			_obj_names.sort()
			obj_names.append(_obj_names)
			for _obj_name in _obj_names:
				ee_names_this_obj = [f.replace(_obj_name, '').replace('pose.txt', '').replace('_', '') for f in
				                     os.listdir(os.path.join(model3d_root, cat_name)) if
				                     f.endswith('pose.txt') and f.startswith(_obj_name)]
				ee_names_this_obj.sort()
				_ee_name.append(ee_names_this_obj)
			ee_names.append(_ee_name)

		for cat_id, cat_name in enumerate(cat_names):
			cat_path = os.path.join(model3d_root, cat_name)
			for obj_id, obj_name in enumerate(obj_names[cat_id]):
				for _, ee_name in enumerate(ee_names[cat_id][obj_id]):
					if 'head2' in ee_name:
						continue
					if is_head_exist(cat_name, obj_name, ee_name, model3d_root):
						stl_path = os.path.join(model3d_root, cat_name, f"{obj_name}.stl")
						obj_pcd = load_pcd_from_stl(stl_path)
						pose_file_name = f"{obj_name}_{ee_name}_pose.txt"
						ee_path = os.path.join(cat_path, pose_file_name)
						ee_RT = np.loadtxt(ee_path, delimiter=',')
						if random_scale_factor > 0.0:
							scale_this = np.random.uniform(1 - random_scale_factor, 1 + random_scale_factor)
							ee_RT[:3, 3] *= scale_this
							obj_pcd *= scale_this
						ee_trans = ee_RT[:3, 3]
						ee_pcd = extract_ee_pcd_by_distance(
							obj_pcd=obj_pcd,
							ee_trans=ee_trans,
							distance_threshold=distance_threshold[f"{cat_name}_{ee_name}"],
						)
						ee_pcd, _ = pcd_downsampling(ee_pcd, 500)
						if 'norm' in pcd_foler_name:
							ee_pcd = normalize_ee_pcd(ee_pcd, ee_RT)
						print(f"cat_name: {cat_name}, obj_name: {obj_name}, ee_name: {ee_name}")
						save_folder = os.path.join(pcd_tmp_folder, cat_name)
						if not os.path.exists(save_folder):
							os.makedirs(save_folder, exist_ok=True)
						np.save(os.path.join(save_folder, f"{obj_name}_{ee_name}.npy"), ee_pcd)
						print(f"save to {os.path.join(save_folder, f'{obj_name}_{ee_name}.npy')}")

if __name__ == '__main__':
	overwrite_pcd = False
	overwrite_distance = False
	overwrite_vis = False
	dist_thrs = 0.03
	pcd_num = 300
	radmoize_scale = False
	random_scale_factor = 0.2  # 20% random scale
	distance_threshold = {
		'hammer_grip_grip': dist_thrs/0.3,
		'hammer_grip_head1': dist_thrs/0.3,
		'screwdriver_head1': dist_thrs/0.2,
		'wrench_head1': dist_thrs/0.2,
	}
	model3d_root = '/dataSSD/1wang/dataspace/Dataset3DModel'
	extract_ee_pcd(model3d_root, distance_threshold, overwrite_pcd, random_scale_factor)
	compute_distance(model3d_root, overwrite_distance)
	visualize_result(model3d_root, overwrite_vis)
	draw_mean_var()







