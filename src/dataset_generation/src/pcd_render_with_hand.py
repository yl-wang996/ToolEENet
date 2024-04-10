import os
# you should use it for tamsgpu4, otherwise the cude error will occur
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import yaml
import PIL.Image as Image
from PIL import ImageDraw

import numpy as np
import open3d as o3d
import rospy
from isaacgym import gymapi
from isaacgym import gymtorch

from tqdm import tqdm

from urdf_generation import URDFLoader
from utils.transform_utils import GymUtil, TfUtils


def random_color():
    return np.random.random(3)


class PCD_Render():
    def __init__(self, cfg, cat_name=None, obj_name=None):
        self.hand_actor_handles = []
        self.obj_actor_handles = []
        self.cam_handles = []
        self.cam_pos = []
        self.cam_lookat = []
        self.envs = []
        self.obj_scales = []
        self.randomized_tz = []


        self.cfg = cfg
        self.headless = self.cfg['sim']['headless']
        self.obj_asset_root = self.cfg["sim"]["obj_asset_root"]
        self.env_num = self.cfg['env']['num_envs']
        self.headless = self.cfg["sim"]["headless"]
        self.env_spacing = self.cfg["env"]["env_spacing"]
        self.save_count = 0
        self.env_per_row = int(self.cfg["env"]["env_per_row"])
        self.ee_name_map = self.cfg["ee_name_map"]
        self.cam_pos_candidate = self.get_cam_pose_candidates()
        self.joint_name_map = self.load_joint_name_map()

        self.urdf_loader = URDFLoader(asset_root_path=self.obj_asset_root)

        self.cat_name = cat_name
        self.obj_name = obj_name

        # initialize gym -----------------------------------------------------------------------------------------------
        self.gym = gymapi.acquire_gym()

        # parse arguments
        self.sim_params = self.setup_sim_params()
        # create sim
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # create viewer
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise ValueError('*** Failed to create viewer')

        self.add_ground()
        self.create_envs()
        self.attach_cameras()

        self.init_all_hand_dof_states()
        self.init_viewer_pose()

        self.hand_actor_ids = self.get_hand_actor_index()
        self.obj_actor_ids = self.get_obj_actor_index()

        # self.randomize_all_camera_pose()
        assert os.path.exists(os.path.join(self.obj_asset_root)), f"No obj_asset_root"
        assert os.path.exists(os.path.join(self.obj_asset_root, self.cat_name)), f"No category"
        self.gym.prepare_sim(self.sim)
        self.acquire_tensors()

    def get_hand_actor_index(self):
        hand_ids = []
        for idx, _ in enumerate(self.envs):
            hand_id = self.gym.get_actor_index(self.envs[idx], self.hand_actor_handles[idx], gymapi.DOMAIN_SIM)
            hand_ids.append(hand_id)
        return hand_ids

    def get_obj_actor_index(self):
        obj_ids = []
        for idx, _ in enumerate(self.envs):
            hand_id = self.gym.get_actor_index(self.envs[idx], self.obj_actor_handles[idx], gymapi.DOMAIN_SIM)
            obj_ids.append(hand_id)
        return obj_ids

    def create_envs(self):
        for index in tqdm(range(self.env_num), 'creating envs'):
            # set up the env grid parameters
            env_lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
            env_upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing * 2)
            # create environment
            env = self.gym.create_env(self.sim, env_lower, env_upper, self.env_per_row)
            self.envs.append(env)
            # Load the object
            self.load_object(env=env, index=index)
            self.load_shadowhand_asset(
                env=env,
                index=index
            )

    def get_cam_pose_candidates(self):
        cam_cfg = self.cfg['env']['camera']
        lookat = cam_cfg['lookat']

        start_x = lookat[0] + cam_cfg['randomize']['offset_limit_x'][0]
        end_x = lookat[0] + cam_cfg['randomize']['offset_limit_x'][1]
        x_list = np.array([])
        x_list = np.append(x_list, np.linspace(start_x, end_x, num=50))
        x_list = np.append(x_list, np.linspace(-end_x, -start_x, num=50))

        start_y = lookat[1] + cam_cfg['randomize']['offset_limit_y'][0]
        end_y = lookat[1] + cam_cfg['randomize']['offset_limit_y'][1]
        y_list = np.array([])
        y_list = np.append(y_list, np.linspace(start_y, end_y, num=50))
        y_list = np.append(y_list, np.linspace(-end_y, -start_y, num=50))

        start_z = cam_cfg['randomize']['limit_z'][0]
        end_z = cam_cfg['randomize']['limit_z'][1]
        z_list = np.array([])
        z_list = np.append(z_list, np.linspace(start_z, end_z, num=50))
        return [x_list,y_list,z_list]

    def get_cam_pose(self):
        cam_cfg = self.cfg['env']['camera']
        lookat = np.asarray(cam_cfg['lookat'])
        if not cam_cfg['randomize']['is_randomize']:
            return cam_cfg['init_location'], lookat
        else:
            if self.cam_pos_candidate is None:
                self.cam_pos_candidate = self.get_cam_pose_candidates()
            x = np.random.choice(self.cam_pos_candidate[0])
            y = np.random.choice(self.cam_pos_candidate[1])
            z = np.random.choice(self.cam_pos_candidate[2])
            position = np.array([x,y,z])
            return position, lookat

    def randomize_camera_pose(self):
        self.cam_pos = []
        self.cam_lookat = []
        for idx, _ in enumerate(self.envs):
            position, lookat = self.get_cam_pose()
            self.cam_pos.append(position)
            self.cam_lookat.append(lookat)
            self.gym.set_camera_location(
                self.cam_handles[idx],
                self.envs[idx],
                gymapi.Vec3(position[0], position[1], position[2]),
                gymapi.Vec3(lookat[0], lookat[1], lookat[2])
                )

    def get_cam_view_transform(self, idx):
        # eye_pos = np.asarray([np.random.random(), np.random.random(), np.random.random() + 1])
        eye_pos = np.asarray(self.cam_pos[idx])
        targget_pos = np.asarray(self.cam_lookat[idx])

        init_M = TfUtils.compose_tf_M(
            trans=eye_pos,
            angles=np.array([90, 0, 90]) / 180 * np.pi
        )

        forward_xy = np.array([eye_pos[0] - targget_pos[0], eye_pos[1] - targget_pos[1]])
        forward_xy_norm = np.linalg.norm(forward_xy)
        if forward_xy_norm != 0:
            forward_xy = forward_xy / forward_xy_norm

        rotate_y = np.arccos(np.dot(np.array([1, 0]), forward_xy))
        if eye_pos[1] < targget_pos[1]:
            rotate_y = -rotate_y
        rotate_y_M = TfUtils.compose_tf_M(
            trans=np.zeros(3),
            angles=np.array([0, rotate_y, 0]),
        )
        init_M = np.dot(init_M, rotate_y_M)

        forward_xz = np.array(
            [eye_pos[0] / np.cos(rotate_y) - targget_pos[0] / np.cos(rotate_y), eye_pos[2] - targget_pos[2]])
        forward_xz_norm = np.linalg.norm(forward_xz)
        if forward_xz_norm != 0:
            forward_xz = forward_xz / forward_xz_norm
        rotate_x = np.arccos(np.dot(np.array([1, 0]), forward_xz))
        if eye_pos[2] < targget_pos[2]:
            rotate_x = -rotate_x
        rotate_x_M = TfUtils.compose_tf_M(
            trans=np.zeros(3),
            angles=np.array([-rotate_x, 0, 0]),
        )
        init_M = np.dot(init_M, rotate_x_M)

        view_transform = init_M
        return view_transform

    def attach_cameras(self):
        for idx, env in enumerate(self.envs):
            # Camera properties
            cam_props = gymapi.CameraProperties()
            cam_props.width = self.cfg['env']['camera']['width']
            cam_props.height = self.cfg['env']['camera']['height']
            cam_props.enable_tensors = True
            cam_handle = self.gym.create_camera_sensor(env, cam_props)
            self.cam_handles.append(cam_handle)

        location, lookat = self.get_cam_pose()
        for idx, cam_handle in enumerate(self.cam_handles):
            self.gym.set_camera_location(
                cam_handle,
                self.envs[idx],
                gymapi.Vec3(location[0], location[1], location[2]),
                gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            )
        if not self.headless:
            self.gym.viewer_camera_look_at(
                self.viewer,  # viewer
                self.envs[0],  # env
                gymapi.Vec3(location[0], location[1], location[2]),  # camera location, z top
                gymapi.Vec3(lookat[0], lookat[1], lookat[2])  # camera target
            )

    def init_viewer_pose(self):
        # the initial camera view of the GUI viewer
        if not self.headless:
            location = self.cfg['env']['camera']['init_location']
            lookat = self.cfg['env']['camera']['lookat']
            self.gym.viewer_camera_look_at(
                self.viewer,  # viewer
                self.envs[0],  # env
                gymapi.Vec3(location[0], location[1], location[2]),  # camera location, z top
                gymapi.Vec3(lookat[0], lookat[1], lookat[2])  # camera target
            )

    def setup_sim_params(self):
        # configure sim -----------------------------------------------------------------------------------------------
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        # since we use the PhysX GPU plugin, we need to use PhysX compatible parameters
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = self.cfg['sim']['num_threads']
        assert self.cfg['sim']['z_top'], 'z_top is required'
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        return sim_params

    def add_ground(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.segmentation_id = 0
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def load_object(self, env, index):
        # Load the object
        asset_options = gymapi.AssetOptions()
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.disable_gravity = True
        # create a new urdf for the object with random color
        if self.cfg['object_randomization']['randomize_color']:
            rgba = None
        else:
            rgba = [0.5, 0.5, 0.5, 1]
        asset_root_path, object_asset_file = self.urdf_loader.create_urdf(
            obj_name=self.obj_name,
            category=self.cat_name,
            rgba=rgba
        )
        object_asset = self.gym.load_asset(self.sim, asset_root_path, object_asset_file, asset_options)
        # Place 3 actors in the environment
        rx, ry, rz = self.cfg['pose_init']['object']['r']
        rx, ry, rz = GymUtil.degree_to_radian(rx, ry, rz)
        r = GymUtil.xyz_to_quaternion(rx, ry, rz)
        r = GymUtil.arrary_to_Quat(r)
        tx, ty, tz = self.cfg['pose_init']['object']['t']
        p = gymapi.Vec3(tx, ty, tz)
        actor_handle = self.gym.create_actor(
            env,
            object_asset,
            gymapi.Transform(r=r, p=p),
            'target_obj',
            index,  # collision group
            self.cfg['env']['collision_filter'],
            # bitwise filter for elements in the same collisionGroup to mask off collision
            segmentationId=int(self.cfg["env"]["object_seg_id"]))

        if self.cfg['object_randomization']['is_randomized']:
            # convert unit from mm to m
            standard_scale = 0.001 * self.cfg['object_randomization']['standard_scale'][self.cat_name]
            random_scale = self.cfg['object_randomization']['random_scale_limit']*(np.random.random() * 2 - 1) + 1
            random_scale = random_scale * standard_scale
            resize_flag = self.gym.set_actor_scale(env, actor_handle, random_scale)

            if resize_flag == False:
                print("3D model Resize Failed")
                return None, None

            rnd_tz = (np.random.random() * 2 - 1) * self.cfg['object_randomization']['random_tz_limit']
            body_state = self.gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
            body_state['pose']["p"]["z"] += rnd_tz
            self.gym.set_actor_rigid_body_states(env, actor_handle, body_state, gymapi.STATE_POS)

        else:
            random_scale = 1
            rnd_tz = 0
        self.randomized_tz.append(rnd_tz)
        self.obj_scales.append(random_scale)
        self.obj_actor_handles.append(actor_handle)

    def randomize_camera_location(self, env, cam_handle):
        self.gym.set_camera_location(
            cam_handle,
            env,)

    # def random_pose(self, env_idx, mode):
    #     assert mode in ['pos', 'rot', 'both', 'none']
    #
    #     # initial hand pos
    #     obj_init_pos = np.asarray(self.cfg['pose_init']['object']['t'])
    #     obj_init_pos[2] += self.randomized_tz[env_idx]
    #     # initial obj rot
    #     obj_angle_init = np.asarray(self.cfg['pose_init']['object']['r'])/180*np.pi
    #
    #     # initial object pos
    #     hand_init_pos = np.asarray(self.cfg['pose_init']['hand']['t'])
    #     # initial hand rot
    #     hand_angle_init = np.asarray(self.cfg['pose_init']['hand']['r'])/180*np.pi  # euler angle in radius
    #
    #     if mode == 'none':
    #         return hand_init_pos, TfUtils.anglexyz_to_quaternion(hand_angle_init), obj_init_pos, TfUtils.anglexyz_to_quaternion(obj_angle_init)
    #
    #     # get angle offset
    #     if mode in ['rot', 'both']:
    #         anglexyz_offset = (np.random.random(3) * 2 - 1) * ((self.cfg['object_randomization']['offset_angle_limit']/180)*np.pi)
    #     else:
    #         anglexyz_offset = np.zeros(3)
    #
    #     if mode in ['pos', 'both']:
    #         pos_offset = self.cfg['object_randomization']['offset_pos_limit']*(np.random.random(3) * 2 - 1)
    #     else:
    #         pos_offset = np.zeros(3)
    #
    #     random_tf_M_trans = TfUtils.compose_tf_M(
    #         trans=pos_offset,
    #         angles=np.zeros(3),
    #     )
    #
    #     random_tf_M_rot = TfUtils.compose_tf_M(
    #         trans=np.zeros(3),
    #         angles=anglexyz_offset,
    #     )
    #
    #     hand_tf_M_init = TfUtils.compose_tf_M(
    #         trans=hand_init_pos,
    #         angles=hand_angle_init
    #     )
    #
    #     obj_tf_M_init = TfUtils.compose_tf_M(
    #         trans=obj_init_pos,
    #         angles=obj_angle_init
    #     )
    #
    #     # transform the hand and obj with the same matrix
    #     hand_tf_M = TfUtils.concat_tf_M([random_tf_M_trans, hand_tf_M_init])
    #     obj_tf_M = TfUtils.concat_tf_M([random_tf_M_trans, obj_tf_M_init])
    #     hand_pos, _ = TfUtils.decompose_tf_M(hand_tf_M)
    #     obj_pos, _ = TfUtils.decompose_tf_M(obj_tf_M)
    #
    #     # transform the hand and obj with the same matrix
    #     hand_tf_M = TfUtils.concat_tf_M([random_tf_M_rot, hand_tf_M_init])
    #     obj_tf_M = TfUtils.concat_tf_M([random_tf_M_rot, obj_tf_M_init])
    #     _, hand_quat = TfUtils.decompose_tf_M(hand_tf_M)
    #     _, obj_quat = TfUtils.decompose_tf_M(obj_tf_M)
    #
    #     return hand_pos, hand_quat, obj_pos, obj_quat






    # def setup_body_state(self, env, actor_handle, pos, quat):
    #     body_state = self.gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
    #     body_state["pose"]["p"]["x"] = pos.x
    #     body_state["pose"]["p"]["y"] = pos.y
    #     body_state["pose"]["p"]["z"] = pos.z
    #     body_state["pose"]["r"]["x"] = quat.x
    #     body_state["pose"]["r"]["y"] = quat.y
    #     body_state["pose"]["r"]["z"] = quat.z
    #     body_state["pose"]["r"]["w"] = quat.w
    #     self.gym.set_actor_rigid_body_states(env, actor_handle, body_state, gymapi.STATE_POS)



    # def randomize_all_object_hand_pose(self):
    #     # hand_root_tensor = gymtorch.wrap_tensor(self.root_tensor)[self.shand_actor_ids]
    #     # obj_root_tensor = gymtorch.wrap_tensor(self.root_tensor)[self.obj_actor_ids]
    #     root_tensor_this = self.root_tensor.clone()
    #     for env_idx, _ in enumerate(self.envs):
    #
    #         hand_pos, hand_quat, obj_pos, obj_quat = self.random_pose(env_idx, mode='both')
    #         hand_quat = torch.from_numpy(hand_quat)
    #         hand_pos = torch.from_numpy(hand_pos)
    #         obj_quat = torch.from_numpy(obj_quat)
    #         obj_pos = torch.from_numpy(obj_pos)
    #
    #
    #         # root_positions = root_tensor[:, 0:3]
    #         # root_orientations = root_tensor[:, 3:7]
    #         # root_linvels = root_tensor[:, 7:10]
    #         # root_angvels = root_tensor[:, 10:13]z
    #
    #         root_tensor_this[self.hand_actor_ids[env_idx]][:3] = hand_pos
    #         root_tensor_this[self.obj_actor_ids[env_idx]][:3] = obj_pos
    #         root_tensor_this[self.hand_actor_ids[env_idx]][3:7] = hand_quat
    #         root_tensor_this[self.obj_actor_ids[env_idx]][3:7] = obj_quat
    #
    #     signal = self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_tensor_this))
    #     assert signal, "setting actor pose failed!"


    def load_shadowhand_asset(self, env, index):
        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0
        asset_options.angular_damping = 0
        asset_options.linear_damping = 0
        asset_options.armature = 0.00
        asset_options.min_particle_mass = 1e-20

        hand_asset = self.gym.load_asset(
            self.sim,
            self.cfg['shadowhand']['asset_root'],
            self.cfg['shadowhand']['hand_asset'],
            asset_options)
        rx, ry, rz = self.cfg['pose_init']['hand']['r']
        rx, ry, rz = GymUtil.degree_to_radian(rx, ry, rz)
        r = GymUtil.xyz_to_quaternion(rx, ry, rz)
        r = GymUtil.arrary_to_Quat(r)
        tx, ty, tz = self.cfg['pose_init']['hand']['t']
        p = gymapi.Vec3(tx, ty, tz)
        actor_handle = self.gym.create_actor(
            env,
            hand_asset,
            gymapi.Transform(r=r, p=p),
            "Shadowhand",
            index,
            self.cfg['env']['collision_filter'],
            self.cfg['env']['hand_seg_id']
        )
        self.hand_actor_handles.append(actor_handle)

    def del_sim(self):
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def depth_to_pointcloud(self, depth_buffer, rgb_buffer, seg_buffer, seg_id, camera_proj_matrix, width, height):
        fu = 2 / camera_proj_matrix[0, 0]
        fv = 2 / camera_proj_matrix [1, 1]
        centerU = width / 2
        centerV = height / 2

        u = range(0, rgb_buffer.shape[1])
        v = range(0, rgb_buffer.shape[0])

        u, v = np.meshgrid(u, v)
        u = u.astype(float)
        v = v.astype(float)

        Z = depth_buffer
        X = -(u - centerU) / width * Z * fu
        Y = (v - centerV) / height * Z * fv

        Z = Z.flatten()
        depth_valid = Z > -10001
        seg_valid = seg_buffer.flatten()==seg_id
        valid = np.logical_and(depth_valid, seg_valid)
        X = X.flatten()
        Y = Y.flatten()

        position = np.vstack((X, Y, Z, np.ones(len(X))))[:, valid].T
        colors = rgb_buffer.reshape((-1,3))[valid]

        points = position[:, 0:3]

        return points, colors

    def ee_pose_to_cam_view(self, ee_pose, cam_RT):
        ee_pose = np.asarray(ee_pose)
        cam_RT = np.linalg.inv(np.asarray(cam_RT))
        ee_pose = cam_RT @ ee_pose
        return ee_pose
    def get_observation(self, env_idx):
        env = self.envs[env_idx]
        cam_handle = self.cam_handles[env_idx]
        cam_width = self.cfg['env']['camera']['width']
        cam_height = self.cfg['env']['camera']['height']

        seg_buffer = gymtorch.wrap_tensor(
            self.gym.get_camera_image_gpu_tensor(
                self.sim,
                env,
                cam_handle,
                gymapi.IMAGE_SEGMENTATION
            )
        ).clone().detach().cpu().numpy()

        depth_buffer =  gymtorch.wrap_tensor(
            self.gym.get_camera_image_gpu_tensor(
                self.sim, env, cam_handle, gymapi.IMAGE_DEPTH
            )
        ).clone().detach().cpu().numpy()
        depth_buffer[seg_buffer == 0] = -10001

        rgb_buffer = gymtorch.wrap_tensor(
            self.gym.get_camera_image_gpu_tensor(
                self.sim, env, cam_handle, gymapi.IMAGE_COLOR
            )
        ).clone().detach().cpu().numpy()
        rgb_buffer = np.resize(rgb_buffer, (cam_height, cam_width, 4))
        rgb_buffer = rgb_buffer[:, :, :3]  # remove alpha channel


        # the camera projection matrix for project the pixels to 3D points under the camera view
        projection_matrix = np.matrix(self.gym.get_camera_proj_matrix(self.sim, self.envs[env_idx], self.cam_handles[env_idx]))
        # the camera view matrix for transform the point cloud from camera view to world view, or verse visa
        view_matrix = np.matrix(self.gym.get_camera_view_matrix(self.sim, self.envs[env_idx], self.cam_handles[env_idx]))
        # camera transform based on the base of each env
        cam_transform = self.get_cam_view_transform(env_idx)
        obj_points, obj_colors = self.depth_to_pointcloud(
            depth_buffer=depth_buffer,
            rgb_buffer=rgb_buffer,
            seg_buffer=seg_buffer,
            seg_id=self.cfg['env']['object_seg_id'],
            camera_proj_matrix=projection_matrix,
            width=cam_width,
            height=cam_height,
        )

        hand_points, hand_colors = self.depth_to_pointcloud(
            depth_buffer=depth_buffer,
            rgb_buffer=rgb_buffer,
            seg_buffer=seg_buffer,
            seg_id=self.cfg['env']['object_seg_id'],
            camera_proj_matrix=projection_matrix,
            width=self.cfg['env']['camera']['width'],
            height=self.cfg['env']['camera']['height'],
        )

        env_base = self.get_env_base_pose(env_idx)

        # get the pose of the object, (x,y,z,w) for quaternion, (x,y,z) for position
        obj_pos = self.root_tensor[self.obj_actor_ids[env_idx]][:3].clone().detach().cpu().numpy()
        obj_quat = self.root_tensor[self.obj_actor_ids[env_idx]][3:7].clone().detach().cpu().numpy()
        obj_M = TfUtils.compose_tf_M(
            trans=obj_pos,
            quat=obj_quat,
        )

        pose_info = {
            'ee_pose':{},
        }

        for ee_name in self.ee_name_map[self.cat_name]:
            ee_pose_file_path = os.path.join(self.obj_asset_root, self.cat_name, f"{self.obj_name}_{ee_name}_pose.txt")
            if os.path.exists(ee_pose_file_path):
                pose_M = self.get_ee_init_pose(
                    ee_pose_file_path=ee_pose_file_path,
                    obj_M=obj_M,
                    idx=env_idx
                )
                # pose_M = self.ee_pose_to_cam_view(ee_pose=pose_M, cam_RT=cam_RT)
                pose_info['ee_pose'][ee_name] = pose_M.tolist()

        meta_info = {
            'camera':{
                'view_matrix': np.asarray(view_matrix).tolist(),
                'projection_matrix': np.asarray(projection_matrix).tolist(),
                'transform': np.asarray(cam_transform).tolist(),
                'width': int(cam_width),
                'height': int(cam_height)
            },
            'object':{
                'scale': float(self.obj_scales[env_idx]),
                'seg_id': int(self.cfg['env']['object_seg_id'])
            },
            'hand':{
                'seg_id': int(self.cfg['env']['hand_seg_id'])
            },
            'env_base':np.asarray(env_base).tolist()
        }

        ee_points, ee_poses, obj_pose = self.to_cam_view(
            ee_poses=pose_info['ee_pose'],
            obj_pose=obj_M,
            cam_transform=cam_transform,
            projection_matrix=projection_matrix,
        )
        pose_info['ee_point'] = ee_points
        pose_info['obj_pose'] = obj_pose
        affordance_seg = self.get_affordance_seg(ee_points=ee_points, seg_buffer=seg_buffer)
        return rgb_buffer, depth_buffer, seg_buffer, affordance_seg, obj_points, obj_colors, hand_points, hand_colors, pose_info, meta_info

    def get_ee_init_pose(self, ee_pose_file_path, idx, obj_M):
        # scale the ee map
        ee_M = np.loadtxt(ee_pose_file_path, delimiter=",")
        ee_trans, ee_quat = TfUtils.decompose_tf_M(ee_M)
        scale = self.obj_scales[idx]
        ee_trans = ee_trans*scale
        ee_M = TfUtils.compose_tf_M(trans=ee_trans, quat=ee_quat)

        ee_pose_M = np.dot(obj_M, ee_M)
        return ee_pose_M

    def load_joint_name_map(self):
        yml_path = os.path.join(self.cfg['shadowhand']['asset_root'], self.cfg['shadowhand']['joint_name_map'])
        with open(yml_path, 'r') as f:
            joint_name_map = yaml.load(f.read(), Loader=yaml.FullLoader)
        return joint_name_map

    def dump_joint_name_map(self, env, actor_handle):
        yml_path = os.path.join(self.cfg['shadowhand']['asset_root'], self.cfg['shadowhand']['joint_name_map'])
        joint_name_map = self.gym.get_actor_joint_dict(env, actor_handle)
        with open(yml_path, 'w') as f:
            yaml.dump(joint_name_map, f)

    def dump_hand_joint_value(self, env, hand_handle, yaml_name):
        data = {}
        dof_states = self.gym.get_actor_dof_states(env, hand_handle, gymapi.STATE_POS)
        for name, index in self.joint_name_map.items():
            data[name] = float(dof_states[index][0])
        yaml_path = os.path.join(self.cfg['shadowhand']['asset_root'], yaml_name)
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)

    def get_env_base_trans(self,env_idx):
        return np.asarray(
            [self.env_spacing * 2 * (env_idx % self.env_per_row),
             self.env_spacing * 2 * (env_idx // self.env_per_row),
             0]
        )

    def init_all_hand_dof_states(self):
        for idx, _ in enumerate(self.envs):
            grasp_states = self.gym.get_actor_dof_states(self.envs[idx], self.hand_actor_handles[idx], gymapi.STATE_POS)
            with open(os.path.join(self.cfg['shadowhand']['asset_root'], self.cfg['shadowhand']['grasp_pose']), 'r') as f:
                grasp_joint_value = yaml.load(f.read(), Loader=yaml.FullLoader)
            for key, value in grasp_joint_value.items():
                grasp_states[self.joint_name_map[key]][0] = value
            success = self.gym.set_actor_dof_states(self.envs[idx], self.hand_actor_handles[idx], grasp_states, gymapi.STATE_POS)
            if not success:
                print(f"Error setting for init grasp pose:{success}, env:{idx}")

    def get_env_base_pose(self, env_idx):
        env_base_pos = np.asarray(
            [self.env_spacing * 2 * (env_idx % self.env_per_row),
             self.env_spacing * 2 * (env_idx // self.env_per_row),
             0])
        return env_base_pos

    def get_max_id(self):
        folder_path = os.path.join(self.cfg['dataset']['save_path'], self.cat_name, self.obj_name)
        if os.path.exists(folder_path) == False:
            os.makedirs(folder_path)
        file_id_list = [int(f.split('.')[0].split('_')[-1]) for f in os.listdir(folder_path)
                        if f.endswith(".pcd")]
        if len(file_id_list) == 0:
            return None
        else:
            max_id = max(file_id_list)
            return max_id

    def to_cam_view(self, ee_poses, obj_pose, cam_transform, projection_matrix):

        ee_poses_copy = ee_poses.copy()
        # view_pose = self.camera_pose_to_view_pose(cam_pose=cam_transform)
        view_pose = cam_transform
        for ee_name, ee_pose in ee_poses_copy.items():
            ee_pose = np.asarray(ee_pose)
            ee_pose = np.linalg.inv(view_pose) @ ee_pose
            ee_poses[ee_name] = ee_pose.tolist()

        obj_pose_new = np.linalg.inv(view_pose) @ obj_pose
        ee_points = self.ee_points_to_image_view(projection_matrix, ee_poses)
        return ee_points, ee_poses, obj_pose_new.tolist()

    def get_circle_seg(self, center, radius=100):
        image_size = (self.cfg['env']['camera']['width'], self.cfg['env']['camera']['height'])
        image = Image.new("L", image_size, "gray")

        # Create a draw object
        draw = ImageDraw.Draw(image)
        # Set the color of the circle (in grayscale, 0 is black, 255 is white)
        circle_color = 255  # Gray color

        # Draw a filled circle
        draw.ellipse([(center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius)],
                     fill=255)

        seg = np.array(image)==circle_color
        return seg



    def get_affordance_seg(self, ee_points, seg_buffer):
        ee_seg = np.zeros_like(seg_buffer)
        obj_seg = seg_buffer == self.cfg['env']['object_seg_id']
        hand_seg = seg_buffer == self.cfg['env']['hand_seg_id']
        ee_seg[obj_seg] = self.cfg['affordance_seg_id_map']['object']
        ee_seg[hand_seg] = self.cfg['affordance_seg_id_map']['hand']
        for ee_name, ee_point in ee_points.items():
            u, v = ee_point
            seg_1 = self.get_circle_seg(center=(u, v), radius=60)
            seg_ee = np.logical_and(seg_1, obj_seg)
            ee_seg[seg_ee] = self.cfg['affordance_seg_id_map'][self.cat_name][ee_name]
        return ee_seg

    def ee_points_to_image_view(self, projection_matrix, ee_poses):
        fu = 2 / projection_matrix[0, 0]
        fv = 2 / projection_matrix[1, 1]

        width = self.cfg['env']['camera']['width']
        height = self.cfg['env']['camera']['height']

        ee_points_new = {}
        centerU = width / 2
        centerV = height / 2
        for ee_name, ee_pose in ee_poses.items():
            x = ee_pose[0][3]
            y = ee_pose[1][3]
            z = ee_pose[2][3]
            u = int(-(x / fu / z * width) + centerU)
            v = int((y / fv / z * height) + centerV)
            ee_points_new[ee_name] = [u, v]
        return ee_points_new

    def camera_pose_to_view_pose(self, cam_pose):
        view_pose = np.asarray(
            cam_pose @ TfUtils.compose_tf_M(trans=np.asarray([0, 0, 0]), angles=np.asarray([90, 0, -90]) / 180 * np.pi))
        return view_pose

    def observe_and_save(self):
        cat_name = self.cat_name
        obj_name = self.obj_name
        finished = False
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for idx, _ in tqdm(enumerate(self.envs), f'rendering envs once'):
            if self.cfg['dataset']['is_save']:
                if self.is_collectoin_finished():
                    finished = True
                    return finished

            # deproject the depth buffer into a point cloud
            rgb_buffer, depth_buffer, seg_buffer, affordance_seg, obj_points, obj_colors, hand_points, hand_colors, pose_info, meta_info = self.get_observation(idx)
            if obj_points is None:
                # in case the point cloud is not complete
                continue

            if self.cfg['dataset']['is_save']:
                folder_path = os.path.join(self.cfg['dataset']['save_path'], cat_name, obj_name)
                os.makedirs(exist_ok=True, name=folder_path)
                max_id = self.get_max_id()
                if max_id is None:
                    m = 0
                else:
                    m = max_id + 1

                # save the point cloud of object
                obj_pcd = o3d.geometry.PointCloud()
                obj_pcd.points = o3d.utility.Vector3dVector(np.asarray(obj_points))
                if np.max(obj_colors) > 1:
                    obj_colors = obj_colors / 255.0
                obj_pcd.colors = o3d.utility.Vector3dVector(obj_colors)
                file_name =  f"obj_pcd_{cat_name}_{obj_name}_{m:04d}.pcd"
                o3d.io.write_point_cloud(
                    os.path.join(folder_path,file_name), obj_pcd)
                meta_info['obj_pcd_path'] = os.path.join(cat_name, obj_name, file_name)
                # save the point cloud of hand
                hand_pcd = o3d.geometry.PointCloud()
                hand_pcd.points = o3d.utility.Vector3dVector(np.asarray(hand_points))
                if np.max(hand_colors) > 1:
                    hand_colors = hand_colors / 255.0
                hand_pcd.colors = o3d.utility.Vector3dVector(hand_colors)
                file_name = f"hand_pcd_{cat_name}_{obj_name}_{m:04d}.pcd"
                o3d.io.write_point_cloud(
                    os.path.join(folder_path, file_name), hand_pcd)
                meta_info['hand_pcd_path'] = os.path.join(cat_name, obj_name, file_name)
                # save the rgb image
                rgb_image = Image.fromarray(rgb_buffer)
                file_name = f"rgb_{cat_name}_{obj_name}_{m:04d}.png"
                rgb_image.save(os.path.join(folder_path, file_name))
                meta_info['image_path'] = os.path.join(cat_name, obj_name, file_name)
                # save the depth data, -10001 is the invalid depth value, means empty
                file_name = f"depth_{cat_name}_{obj_name}_{m:04d}"
                np.savez_compressed(os.path.join(folder_path, file_name), depth_buffer)
                meta_info['depth_path'] = os.path.join(cat_name, obj_name, file_name)+'.npz'
                # save the seg data, 0 denotes the background, 1 denotes the object, 2 denotes the hand
                file_name = f"seg_{cat_name}_{obj_name}_{m:04d}.png"
                seg_image = Image.fromarray(seg_buffer)
                seg_image.save(fp=os.path.join(folder_path, file_name))
                meta_info['seg_path'] = os.path.join(cat_name, obj_name, file_name)
                # save the affordance seg data
                file_name = f"affordance_seg_{cat_name}_{obj_name}_{m:04d}.png"
                affordance_seg_image = Image.fromarray(affordance_seg)
                affordance_seg_image.save(fp=os.path.join(folder_path, file_name))
                meta_info['affordance_seg_path'] = os.path.join(cat_name, obj_name, file_name)
                # save the pose info
                with open(os.path.join(folder_path, f"pose_{cat_name}_{obj_name}_{m:04d}.yaml"), 'w') as yaml_file:
                    yaml.dump(pose_info, yaml_file, default_flow_style=False)
                # save the meta info
                with open(os.path.join(folder_path, f"meta_{cat_name}_{obj_name}_{m:04d}.yaml"), 'w') as yaml_file:
                    yaml.dump(meta_info, yaml_file, default_flow_style=False)
                print(f"Saved {cat_name}_{obj_name}_{m:04d}")
                self.save_count += 1

        self.gym.end_access_image_tensors(self.sim)
        return finished

    def is_collectoin_finished(self):
        save_path = os.path.join(self.cfg['dataset']['save_path'], self.cat_name, self.obj_name)
        if  os.path.exists(save_path) == False:
            os.makedirs(save_path)
        file_list = [f for f in os.listdir(save_path) if 'obj_pcd' in f]
        data_size = self.cfg['dataset']['total_num_per_obj']

        if len(file_list) >= data_size:
            # print('self.is_collectoin_finished')
            return True
        else:
            return False

    def acquire_tensors(self):
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_tensor = gymtorch.wrap_tensor(_root_tensor)
        self.root_tensor = root_tensor
        # root_positions = root_tensor[:, 0:3]
        # root_orientations = root_tensor[:, 3:7]
        # root_linvels = root_tensor[:, 7:10]
        # root_angvels = root_tensor[:, 10:13]

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_states = dof_states

    def refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

def is_finished(cat_name, obj_name, cfg):
    folder_path = os.path.join(cfg['dataset']['save_path'], cat_name, obj_name)
    if os.path.exists(folder_path) == False:
        os.makedirs(folder_path)
    file_id_list = [int(f.split('.')[0].split('_')[-1]) for f in os.listdir(folder_path)
                    if 'obj_pcd' in f]
    if len(file_id_list) >= cfg['dataset']['total_num_per_obj']:
        return True
    else:
        return False

def generate(cat_name, obj_name, cfg):
    pcd_render = PCD_Render(
        cfg=cfg,
        cat_name=cat_name,
        obj_name=obj_name,
    )

    frame_count = 0
    save_per_frame = pcd_render.cfg['env']['save_per_frame']

    while True:
        # # do somthing here
        if frame_count%save_per_frame==0 and frame_count!=0:
            # dof_tensor = gymtorch.wrap_tensor(pcd_render.dof_states)
            pcd_render.refresh()
            print('randomize_camera_pose')

            # pcd_render.randomize_all_object_hand_pose()
            pcd_render.randomize_camera_pose()

        # step the physics
        pcd_render.gym.simulate(pcd_render.sim)
        pcd_render.gym.fetch_results(pcd_render.sim, True)
        # update graphics
        pcd_render.gym.step_graphics(pcd_render.sim)

        # Update viewer and check for exit conditions
        if not pcd_render.headless:
            if pcd_render.gym.query_viewer_has_closed(pcd_render.viewer):
                break
            pcd_render.gym.draw_viewer(pcd_render.viewer, pcd_render.sim, False)

        if frame_count%save_per_frame==0 and pcd_render.cfg['dataset']['is_save'] and frame_count!=0:
            finished = pcd_render.observe_and_save()
            if finished:
                break

        frame_count = frame_count + 1

# TODO, extract the dataset for the training, to verify
if __name__ == '__main__':
    config_yaml = "/homeL/1wang/workspace/toolee_ws/src/dataset_generation/src/cfg/config.yaml"
    with open(config_yaml, 'r') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    data_root = cfg['sim']['obj_asset_root']
    cat_list = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    cat_list.sort()
    for cat in cat_list:
        print(cat)
        obj_list = [f.replace('.stl', '') for f in os.listdir(os.path.join(data_root, cat)) if f.endswith('.stl')]
        obj_list.sort()
        for obj in obj_list:
            if is_finished(cat_name=cat,obj_name=obj,cfg=cfg):
                print(f"category: {cat}  object:{obj} is finished for collection")
                continue
            print(f'collecting category: {cat}  object:{obj}')
            generate(
                cat_name=cat,
                obj_name=obj,
                cfg=cfg,
            )
            quit()
