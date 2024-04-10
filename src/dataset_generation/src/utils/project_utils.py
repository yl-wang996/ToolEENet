import numpy as np



def project_depth_to_pointscloud(depth_buffer, rgb_buffer, seg_buffer, seg_id, camera_view_matrix, camera_proj_matrix, img_width, img_height):
	vinv = np.linalg.inv(camera_view_matrix)
	fu = 2 / camera_proj_matrix[0, 0]
	fv = 2 / camera_proj_matrix [1, 1]
	centerU = img_width / 2
	centerV = img_height / 2
	
	u = range(0, rgb_buffer.shape[1])
	v = range(0, rgb_buffer.shape[0])
	
	u, v = np.meshgrid(u, v)
	u = u.astype(float)
	v = v.astype(float)
	
	Z = depth_buffer
	X = -(u - centerU) / img_width * Z * fu
	Y = (v - centerV) / img_height * Z * fv
	
	Z = Z.flatten()
	depth_valid = Z > -10001
	seg_valid = seg_buffer.flatten() ==seg_id
	valid = np.logical_and(depth_valid, seg_valid)
	X = X.flatten()
	Y = Y.flatten()
	
	position = np.vstack((X, Y, Z, np.ones(len(X))))[:, valid].T
	colors = rgb_buffer.reshape((-1 ,3))[valid]
	position = position * vinv
	
	points = position[:, 0:3]
	return points, colors

def project_xyz_to_pixel_uv(camera_proj_matrix, camera_view_matrix, points):
	pass