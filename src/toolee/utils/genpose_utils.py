import torch

def get_pose_dim(pose_mode):
    assert pose_mode in ['rot_matrix_symtr','rot_matrix']
    if pose_mode == 'rot_matrix_symtr':
        return 6+3+3
    elif pose_mode == 'rot_matrix':
        return 6+3
    else:
        raise NotImplementedError

class TrainClock(object):
    """ Clock object to track epoch and step during training
    """
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0
    
    # one step
    def tick(self):
        self.minibatch += 1
        self.step += 1
    
    # one epoch
    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']


def merge_results(results_ori, results_new):
    if len(results_ori.keys()) == 0:
        return results_new
    else:
        results = {
            'pred_pose': torch.cat([results_ori['pred_pose'], results_new['pred_pose']], dim=0),
            'gt_pose': torch.cat([results_ori['gt_pose'], results_new['gt_pose']], dim=0),
            'cls_id': torch.cat([results_ori['cls_id'], results_new['cls_id']], dim=0),
            # 'path': results_ori['path'] + results_new['path'],
        }
        return results


