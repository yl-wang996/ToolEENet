import numpy as np
import open3d as o3d
import os

obj_id = "hammer_10"
file_path = "/homeL/1wang/workspace/anno_ee_ws/src/pose_annotation/meshes/Dataset3DModel_v2.0/Hammer_Grip/"

mesh = o3d.io.read_triangle_mesh(os.path.join(file_path, f"{obj_id}.obj"))
print(mesh.get_center())
mesh.scale(1, center=mesh.get_center())
print(np.min(mesh.vertices, axis=0))
print(np.max(mesh.vertices, axis=0))
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

# o3d.visualization.draw_geometries([mesh])

o3d.io.write_triangle_mesh(os.path.join(file_path, f"{obj_id}.stl"), mesh,write_vertex_colors=False,write_ascii=False,write_vertex_normals=True,write_triangle_uvs=False)

