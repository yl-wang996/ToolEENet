import argparse

# the mapping between the affordance name and the affordance id in the affordance segmentation image
affordance_seg_id_map={
    "empty": 0,
    "hand": 1,
    "object": 2,
    "hammer_grip_head1": 3,
    "hammer_grip_grip": 4,
    "screwdriver_head1": 5,
    "wrench_head1": 6,
    "wrench_head2": 7,
}
def get_affordance_id_from_name(cat, ee_name):
    name = f"{cat}_{ee_name}"
    return affordance_seg_id_map[name]
def get_affordance_name_from_id(id):
    for k,v in affordance_seg_id_map.items():
        if v==id:
            return k
    return None
def get_all_ee_seg_names():
    return list(affordance_seg_id_map.keys())[3:]
def get_all_ee_seg_ids():
    return list(affordance_seg_id_map.values())[3:]
def get_affordance_seg_id_map():
    return affordance_seg_id_map

def get_config(show=True):
    parser = argparse.ArgumentParser()
    
    """ dataset """
    # parser.add_argument('--synset_names', nargs='+', default=['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug'])
    parser.add_argument('--data_path', default='/dataSSD/yunlong/dataspace/DatasetToolEE', type=str)
    parser.add_argument('--asset_path', default='/dataSSD/yunlong/dataspace/Dataset3DModel', type=str)
    parser.add_argument('--o2c_pose', default=True, action='store_true')
    parser.add_argument('--batch_size', type=int, default=600)
    parser.add_argument('--eval_batch_size', type=int, default=200)
    parser.add_argument('--pose_mode', type=str, default='rot_matrix')  # rot_matrix_symtr, rot_matrix
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--percentage_data_for_train', type=float, default=1.0) 
    parser.add_argument('--percentage_data_for_val', type=float, default=0.1) # 0.1 for accelerating the testing
    parser.add_argument('--percentage_data_for_test', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--per_obj', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--task_type', type=str, default='ee_pose')  # ee_pose, obj_pose,
    
    
    """ model """
    parser.add_argument('--posenet_mode',  type=str, default='score')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--sampler_mode', nargs='+')
    parser.add_argument('--sampling_steps', type=int)
    parser.add_argument('--sde_mode', type=str, default='ve')
    parser.add_argument('--sigma', type=float, default=25)  # base-sigma for SDE
    parser.add_argument('--likelihood_weighting', default=False, action='store_true')
    parser.add_argument('--regression_head', type=str, default='Rx_Ry_and_T')  # Rx_Ry_and_T, Rx_Ry_and_T_and_Symtr
    parser.add_argument('--pointnet2_params', type=str, default='light')
    parser.add_argument('--pts_encoder', type=str, default='pointnet2')
    
    
    """ training """
    parser.add_argument('--agent_type', type=str, default='score', help='only score')
    parser.add_argument('--pretrained_score_model_path', type=str)

    parser.add_argument('--distillation', default=False, action='store_true')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='ScoreNet')
    parser.add_argument('--log_folder', type=str, default='/dataSSD/yunlong/dataspace/train_logs')
    parser.add_argument('--optimizer',  type=str, default='Adam')
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--repeat_num', type=int, default=20)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--ema_rate', type=float, default=0.999)  # ema force the smooth training, prevent the shaking of the steps
    # ema: mean a weighted average of the current model parameters and the previous parameters, 0.99 is the weight of the previous model parameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--use_pretrain', default=False, action='store_true')
    parser.add_argument('--parallel', default=False, action='store_true')   
    parser.add_argument('--num_gpu', type=int, default=2)
    parser.add_argument('--is_train', default=False, action='store_true')
    
    """ testing """
    parser.add_argument('--eval_set', default='test', type=str)  # test, novel
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--pred', default=False, action='store_true')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--eval_repeat_num', type=int, default=20)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--max_eval_num', type=int, default=10000000)
    parser.add_argument('--img_size', type=int, default=256, help='cropped image size')
    parser.add_argument('--result_dir', type=str, default='', help='result directory')
    parser.add_argument('--T0', type=float, default=1.0)
    
    # cfg = parser.parse_args()
    cfg, _ = parser.parse_known_args()
    
    cfg.cat_name = ['hammer_grip', 'screwdriver', 'wrench']
    cfg.ee_names = ['hammer_grip_head1', 'hammer_grip_grip', 'screwdriver_head1', 'wrench_head1', 'wrench_head2']
    
    if show:
        for k, v in cfg.__dict__.items():
            print(f'{k}: {v}')

    return cfg

