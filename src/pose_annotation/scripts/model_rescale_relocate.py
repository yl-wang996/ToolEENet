import numpy as np
import open3d as o3d
import os

def main():
	data_root = "/homeL/1wang/workspace/anno_ee_ws/src/pose_annotation/meshes/"
	input_dataset_name = "Dataset3DModel_v3.0_unnormalized"
	output_dataset_name = "Dataset3DModel_v3.0_norm_location_scale"
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
			# pose_path = os.path.join(path, file.split(".")[0] + "_head_pose.txt")
			# T = np.loadtxt(pose_path, delimiter=",")
			# mesh.transform(T)
			
			# scale the object to the unit size of the diagonal of the bounding box
			vertices = np.asarray(mesh.vertices)
			print(f"max: {np.max(vertices, axis=0)}")
			x_range = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
			y_range = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
			z_range = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
			scale = np.sqrt(x_range ** 2 + y_range ** 2 + z_range ** 2)
			print(f"scale_before: {scale}")
			mesh.scale(1 / scale, center=mesh.get_center())
			# # move the object to the zero mean center
			vertices = np.asarray(mesh.vertices)
			x_range = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
			y_range = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
			z_range = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
			print(f"scale_after: {np.sqrt(x_range ** 2 + y_range ** 2 + z_range ** 2)}")
			
			cx = (np.max(vertices[:, 0]) + np.min(vertices[:, 0])) / 2
			cy = (np.max(vertices[:, 1]) + np.min(vertices[:, 1])) / 2
			cz = (np.max(vertices[:, 2]) + np.min(vertices[:, 2])) / 2
			center = np.array([cx, cy, cz])
			print(f"center_before: {center}")
			mesh.translate(-center, relative=True)
			print(f"center_after: {mesh.get_center()}")  # [0, 0, 0]
			
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
	main()
	# inspection("/homeL/1wang/workspace/anno_ee_ws/src/pose_annotation/meshes/Dataset3DModel_v3.0/hammer_grip/hammer_01.stl")