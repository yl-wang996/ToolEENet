import numpy as np
import open3d as o3d
import os


def main(mesh_content_path=None):
	data_root = "/homeL/1wang/workspace/anno_ee_ws/src/pose_annotation/meshes/"
	input_dataset_name = "Dataset3DModel_v3.0_norm_location_scale"
	output_dataset_name = "Dataset3DModel_v3.0"
	if mesh_content_path is not None:
		mesh_path = os.path.join(data_root, input_dataset_name, mesh_content_path)
		mesh = o3d.io.read_triangle_mesh(mesh_path)
		norm_pose_path = mesh_path.replace(".stl", "_norm_pose.txt")
		T = np.loadtxt(norm_pose_path, delimiter=",")
		mesh.transform(T)
		output_stl_path = mesh_path.replace(input_dataset_name, output_dataset_name)
		output_obj_path = output_stl_path.replace('.stl', '.obj')
		o3d.io.write_triangle_mesh(
			output_obj_path,
			mesh,
			write_vertex_colors=False,
			write_triangle_uvs=False
		)
		
		mesh.compute_vertex_normals()
		o3d.io.write_triangle_mesh(
			output_stl_path,
			mesh,
			write_vertex_colors=False,
			write_triangle_uvs=False
		)
		return
	
	cats = os.listdir(os.path.join(data_root, input_dataset_name))
	cat_list = [f for f in cats if os.path.isdir(os.path.join(data_root, input_dataset_name, f))]
	for cat in cat_list:
		cat_path = os.path.join(data_root, input_dataset_name, cat)
		stl_list = [f for f in os.listdir(cat_path) if f.endswith(".stl")]
		stl_list.sort()
		for stl in stl_list:
			stl_path = os.path.join(cat_path, stl)
			target_folder = os.path.join(data_root, output_dataset_name, cat)
			if not os.path.exists(target_folder):
				os.makedirs(target_folder)
			
			print(f"Processing {stl_path}" + "-"*20)
			
			mesh = o3d.io.read_triangle_mesh(stl_path)
			
			# # for reorient the object
			# normalize the orientation of the object by hand annotation
			# hammer_01_norm_pose.txt
			norm_pose_path = os.path.join(data_root, input_dataset_name, cat, stl.split("/")[-1].split(".")[0] + "_norm_pose.txt")
			T = np.loadtxt(norm_pose_path, delimiter=",")
			mesh.transform(T)
			
			o3d.io.write_triangle_mesh(
				os.path.join(target_folder, stl.replace(".stl", ".obj")),
				mesh,
				write_vertex_colors=False,
				write_triangle_uvs=False
			)
			
			mesh.compute_vertex_normals()
			o3d.io.write_triangle_mesh(
				os.path.join(target_folder, stl),
				mesh,
				write_vertex_colors=False,
				write_triangle_uvs=False
			)
def inspection(model_path):
	mesh = o3d.io.read_triangle_mesh(model_path)
	vertices = np.asarray(mesh.vertices)
	print(f"max: {np.max(vertices, axis=0)}")
	print(f"min: {np.min(vertices, axis=0)}")
	print(f"center: {mesh.get_center()}")

if __name__ == '__main__':
	cat = "wrench"
	obj = "wrench_17"
	
	main(mesh_content_path=f"{cat}/{obj}.stl")
	# inspection("/homeL/1wang/workspace/anno_ee_ws/src/pose_annotation/meshes/Dataset3DModel_v3.0/hammer_grip/hammer_01.stl")