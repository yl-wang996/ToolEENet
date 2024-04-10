import functools
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import utils.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from networks.gf_algorithms.score_utils import ExponentialMovingAverage
from networks.gf_algorithms.losses import loss_fn, loss_fn_edm
from networks.gf_algorithms.sde import init_sde
from networks.posenet import GFObjectPose

from utils.genpose_utils import TrainClock
from utils.misc import exists_or_mkdir, average_quaternion_batch
from utils.visualize import create_grid_image, test_time_visulize
from utils.metrics import get_errors, get_rot_matrix


def get_ckpt_and_writer_path(cfg):
    pth_path = os.path.join(cfg.log_folder,f"results/ckpts/{cfg.log_dir}/ckpt_epoch{cfg.model_name}.pth")
    if cfg.use_pretrain and not os.path.exists(pth_path):
        raise Exception(f"{pth_path} is not exist!")
    
    ''' init exp folder and writer '''
    ckpt_path = os.path.join(cfg.log_folder, f'results/ckpts/{cfg.log_dir}')
    writer_path = os.path.join(cfg.log_folder, f'results/logs/{cfg.log_dir}') if cfg.use_pretrain == False else os.path.join(cfg.log_folder, f'results/logs/{cfg.log_dir}_continue')
    
    if cfg.is_train:
        exists_or_mkdir(os.path.join(cfg.log_folder, 'results'))
        exists_or_mkdir(ckpt_path)
        exists_or_mkdir(writer_path)
    return ckpt_path, writer_path


class PoseNet(nn.Module):
    def __init__(self, cfg):
        super(PoseNet, self).__init__()
        
        self.cfg = cfg
        self.pred_symtr = True if 'symtr' in cfg.pose_mode else False
        self.is_testing = False
        self.clock = TrainClock()
        self.pts_feature = False
        self.recorder = {
            'mean_rot_error': 9999,
            'mean_trans_error': 9999,
        }

        # get checkpoint and writer path
        self.model_dir, writer_path = get_ckpt_and_writer_path(self.cfg)
        
        # init writer
        if self.cfg.is_train:
            self.writer = SummaryWriter(writer_path)
        
        # init sde, Stochastic Differential Equation (SDE) for the diffusion process
        self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps, self.T = init_sde(self.cfg.sde_mode)
        self.net = self.build_net()
        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler()
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=self.cfg.ema_rate)

        # init related functions
        if self.cfg.sde_mode == 'edm':
            self.loss_fn = functools.partial(loss_fn_edm, sigma_data=1.4148, P_mean=-1.2, P_std=1.2)
        else:
            self.loss_fn = loss_fn
            
    def is_best(self,rot_err, trans_err):
        if rot_err < self.recorder['mean_rot_error'] and trans_err < self.recorder['mean_trans_error']:
            return True
        else:
            return False
    
    def update_recorder(self, rot_err,trans_err):
        self.recorder['mean_rot_error'] = rot_err
        self.recorder['mean_trans_error'] = trans_err
        
    
    
    # get network and move to device
    def build_net(self):
        net = GFObjectPose(self.cfg, self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps, self.T)
        net = net.to(self.cfg.device)
        if self.cfg.parallel:
            device_ids = list(range(self.cfg.num_gpu))
            net = nn.DataParallel(net, device_ids=device_ids).cuda()
        return net
    
    def set_optimizer(self):
        """set optimizer used in training"""
        params = []
        params = self.net.parameters()            
        self.base_lr = self.cfg.lr
        if self.cfg.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.cfg.lr,
                momentum=0.9,
                weight_decay=1e-4
            )
        elif self.cfg.optimizer == 'Adam':
            optimizer = optim.Adam(params, betas=(0.9, 0.999), eps=1e-8, lr=self.cfg.lr)     
        else:
            raise NotImplementedError
        return optimizer

    def set_scheduler(self):
        """set lr scheduler used in training"""
        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.cfg.lr_decay)
        return scheduler

    def save_ckpt(self, name=None, is_best=False):
        """save checkpoint during training for future restore"""
        best_path = os.path.join(self.model_dir, "best.pth")
        if name is not None:
            save_path = os.path.join(self.model_dir, name)
        else:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
        print("Saving checkpoint epoch {}...".format(self.clock.epoch))

        self.ema.store(self.net.parameters())
        self.ema.copy_to(self.net.parameters())
        if isinstance(self.net, nn.DataParallel):
            model_state_dict = self.net.module.cpu().state_dict()
        else:
            model_state_dict = self.net.cpu().state_dict()
            
        if is_best:
            torch.save({
                'clock': self.clock.make_checkpoint(),
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, best_path)

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)

        self.net.to(self.cfg.device)
        self.ema.restore(self.net.parameters())
        
    def load_ckpt(self, name=None, model_dir=None, model_path=False, load_model_only=False):
        """load checkpoint from saved checkpoint"""
        if not model_path:
            if name == 'latest':
                pass
            elif name == 'best':
                pass
            else:
                name = "ckpt_epoch{}".format(name)

            if model_dir is None:
                load_path = os.path.join(self.model_dir, "{}.pth".format(name))
            else:
                load_path = os.path.join(model_dir, "{}.pth".format(name))
        else:
            load_path = model_dir
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])
        
        if not load_model_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.clock.restore_checkpoint(checkpoint['clock'])

    # avg loss of loss_fn from each batch
    def collect_score_loss(self, data, teacher_model=None, pts_feat_teacher=None):
        '''
        Repeat the forward process for several times and calculate the average loss
        Args:
            data, dict {
                'pts': [bs, c]
                'gt_pose': [bs, pose_dim]
            }
        '''
        gf_loss = 0
        for _ in range(self.cfg.repeat_num):
            gf_loss += self.loss_fn(
                model=self.net,
                data=data,
                marginal_prob_func=self.marginal_prob_fn,
                sde_fn=self.sde_fn,
                likelihood_weighting=self.cfg.likelihood_weighting, 
                teacher_model=teacher_model,
                pts_feat_teacher=pts_feat_teacher
            )
        gf_loss /= self.cfg.repeat_num
        losses = {'gf': gf_loss}        
        return losses

    def collect_ema_loss(self, data):
        '''
        Args:
            data, dict {
                'pts': [bs, c]
                'gt_pose': [bs, pose_dim]
            }
        '''
        self.ema.store(self.net.parameters())
        self.ema.copy_to(self.net.parameters())
        with torch.no_grad():
            ema_loss = 0
            for _ in range(self.cfg.repeat_num):
                # calc score-matching loss
                ema_loss += self.loss_fn(
                    model=self.net,
                    data=data,
                    marginal_prob_func=self.marginal_prob_fn,
                    sde_fn=self.sde_fn,
                    likelihood_weighting=self.cfg.likelihood_weighting
                )
            ema_loss /= self.cfg.repeat_num
        self.ema.restore(self.net.parameters())
        ema_losses = {'ema': ema_loss}        
        return ema_losses        

    def train_score_func(self, data, teacher_model=None):
        """ One step of training """
        self.net.train()
        self.is_testing = False
        # extract the pts_feature from the pointnet++
        data['pts_feat'] = self.net(data, mode='pts_feature')
        with torch.no_grad():
            if teacher_model is not None:
                teacher_model.eval()
            pts_feat_teacher = None if teacher_model is None else teacher_model(data, mode='pts_feature')
        self.pts_feature = True
        gf_losses = self.collect_score_loss(data, teacher_model, pts_feat_teacher)
        
        self.update_network(gf_losses)
        self.record_losses(gf_losses, 'train')
        self.record_lr()
        
        self.ema.update(self.net.parameters())
        if self.cfg.ema_rate > 0 and self.clock.step % 5 == 0:
            ema_losses = self.collect_ema_loss(data)
            self.record_losses(ema_losses, 'train')
        self.pts_feature = False
        return gf_losses
    
    
    def train_func(self, data, pose_samples=None, gf_mode='score', teacher_model=None):
        assert gf_mode == 'score', 'Only support score mode for now.'
        losses = self.train_score_func(data, teacher_model)
        return losses
    
    
    def eval_score_func(self, data, data_mode):
        self.is_testing = True
        self.net.eval()
        self.ema.store(self.net.parameters())
        self.ema.copy_to(self.net.parameters())
        with torch.no_grad():
            # get the pts_feature from the pointnet++
            data['pts_feat'] = self.net(data, mode='pts_feature')
            self.pts_feature = True
            in_process_sample_list = []
            res_list = []
            sampler_mode_list = self.cfg.sampler_mode
            if isinstance(sampler_mode_list, str):
                sampler_mode_list = [sampler_mode_list]
                
            for sampler in sampler_mode_list:
                in_process_sample, res = self.net(data, mode=f'{sampler}_sample')
                in_process_sample_list.append(in_process_sample)
                res_list.append(res)
            
            metrics = []
            for res_item, sampler_item in zip(res_list, sampler_mode_list):
                metric = self.collect_metric(res_item, data['gt_pose'], data['id'])
                metrics.append(metric)
                self.record_metrics(metric, sampler_item, data_mode)
                
            self.visualize_batch(data, res_list, sampler_mode_list, data_mode)
            
            if self.cfg.save_video:
                save_path = self.model_dir.replace('ckpts', 'inference_results')
                save_path = os.path.join(
                    save_path, 
                    data_mode + '_' + sampler_item + '_' + str(self.cfg.sampling_steps),
                    f'epoch_{str(self.clock.epoch)}'
                )
                print('Saving videos and images...')
                test_time_visulize(save_path, data, res, in_process_sample, self.cfg.pose_mode, self.cfg.o2c_pose)
            self.pts_feature = False
        self.ema.restore(self.net.parameters())
        return metrics, sampler_mode_list
    
    
    # def eval_energy_func(self, data, data_mode, pose_samples):
    #     self.is_testing = True
    #     self.net.eval()
    #
    #     with torch.no_grad():
    #         data['pts_feat'] = self.net(data, mode='pts_feature')
    #         self.pts_feature = True
    #
    #         score_losses = self.collect_score_loss(data)
    #         ranking_losses = self.collect_ranking_loss(data, pose_samples)
    #         gf_losses = {**score_losses, **ranking_losses}
    #         for k, v in ranking_losses.items():
    #             if not k == 'item':
    #                 self.writer.add_scalar(f'{data_mode}/loss_{k}', v, self.clock.epoch)
    #
    #     return gf_losses, None


    def eval_func(self, data, data_mode, pose_samples=None, gf_mode='score'):
        assert gf_mode == 'score', 'Only support score mode for now.'
        metrics, sampler_mode_list = self.eval_score_func(data, data_mode)
        return metrics, sampler_mode_list

        
    
    def test_func(self, data, batch_id):
        self.is_testing = True
        self.net.eval()
        
        with torch.no_grad():
            data['pts_feat'] = self.net(data, mode='pts_feature')
            self.pts_feature = True            
            in_process_sample, res = self.net(data, mode=f'{self.cfg.sampler_mode[0]}_sample')
            sampler_item = self.cfg.sampler_mode[0]
            results = {
                'pred_pose': res,
                'gt_pose': data['gt_pose'],
                'cls_id': data['id'],
            }
            metrics = self.collect_metric(res, data['gt_pose'], data['id'])
            if self.cfg.save_video:
                save_path = self.model_dir.replace('ckpts', 'inference_results')
                save_path = os.path.join(
                    save_path, 
                    self.cfg.test_source + '_' + sampler_item + '_' + str(self.cfg.sampling_steps), 
                    f'eval_num_{str(batch_id)}'
                )
                print('Saving videos and images...')
                test_time_visulize(save_path, data, res, in_process_sample, self.cfg.pose_mode, self.cfg.o2c_pose)
            self.pts_feature = False
            
        return metrics, sampler_item, results


    def pred_func(self, data, repeat_num, T0=None):
        """
        
        Args:
            data: batch data
            repeat_num: how much time to repeat the prediction to get the consistent results
            T0:
        Returns:
            pred_pose,
            pred_pose_q_wxyz,
            average_pred_pose_q_wxyz,  averaged pose along the repeat number
            in_process_sample, the intermediate results of the forward(add noise) process

        """
        self.is_testing = True
        self.net.eval()
        
        with torch.no_grad():
            # extract the pts_feature from the pointnet++
            data['pts_feat'] = self.net(data, mode='pts_feature')
            bs = data['pts'].shape[0]
            self.pts_feature = True
            
            ''' Repeat input data, [bs, ...] to [bs*repeat_num, ...] '''
            repeated_data = {}
            for key in data.keys():
                data_shape = [item for item in data[key].shape]
                repeat_list = np.ones(len(data_shape) + 1, dtype=np.int8).tolist()
                repeat_list[1] = repeat_num
                repeated_data[key] = data[key].unsqueeze(1).repeat(repeat_list)
                data_shape[0] = bs*repeat_num
                repeated_data[key] = repeated_data[key].view(data_shape)
                
            ''' Inference '''
            in_process_sample, res = self.net(repeated_data, mode=f'{self.cfg.sampler_mode[0]}_sample', init_x=None, T0=T0)
            pred_pose = res.reshape(bs, repeat_num, -1)
            in_process_sample = in_process_sample.reshape(bs, repeat_num, in_process_sample.shape[1], -1)
            
            self.pts_feature = False

            ''' Calculate the average results '''
            rot_matrix = get_rot_matrix(res[:, :6])
            quat_wxyz = transforms.matrix_to_quaternion(rot_matrix)
            res_q_wxyz = torch.cat((quat_wxyz, res[:, 6:9]), dim=-1)
            pred_pose_q_wxyz = res_q_wxyz.reshape(bs, repeat_num, -1)    # [bs, repeat_num, pose_dim]
            
            average_pred_pose_q_wxyz = torch.zeros((bs, 7)).to(pred_pose_q_wxyz.device) # [bs, 7], 7 = 4(quaternion) + 3(translation)
            average_pred_pose_q_wxyz[:, :4] = average_quaternion_batch(pred_pose_q_wxyz[:, :, :4])
            average_pred_pose_q_wxyz[:, 4:] = torch.mean(pred_pose_q_wxyz[:, :, 4:], dim=1)

            return pred_pose, pred_pose_q_wxyz, average_pred_pose_q_wxyz, in_process_sample

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), 
                max_norm=self.cfg.grad_clip
            )
        self.optimizer.step()


    def update_learning_rate(self):
        """record and update learning rate"""
        # self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        if self.clock.step <= self.cfg.warmup:
            self.optimizer.param_groups[-1]['lr'] = self.base_lr / self.cfg.warmup * self.clock.step
        # elif not self.optimizer.param_groups[-1]['lr'] < self.base_lr / 20.0:
        elif not self.optimizer.param_groups[-1]['lr'] < 1e-4:
            self.scheduler.step()


    def record_losses(self, loss_dict, mode='train'):
        """record loss to tensorboard"""
        losses_values = {k: v.item() for k, v in loss_dict.items()}
        for k, v in losses_values.items():
            self.writer.add_scalar(f'{mode}/{k}', v, self.clock.step)
    
    
    def record_metrics(self, metric, sampler_mode, mode='val'):
        """record metric to tensorboard"""
        rot_error = metric['rot_error']
        trans_error = metric['trans_error']
        
        for k, v in rot_error.items():
            if not k == 'item':
                self.writer.add_scalar(f'{mode}/{sampler_mode}_{k}_rot_error', v, self.clock.epoch)
        for k, v in trans_error.items():
            if not k == 'item':
                self.writer.add_scalar(f'{mode}/{sampler_mode}_{k}_trans_error', v, self.clock.epoch)

    def record_lr(self):
        self.writer.add_scalar('learing_rate', self.optimizer.param_groups[0]['lr'], self.clock.step)
       
    def visualize_batch(self, data, res, sampler_mode, mode):
        """write visualization results to tensorboard writer"""
        for res_item, sampler_item in zip(res, sampler_mode):
            pts = torch.cat((data['pts'], data['pts_color']), dim=2)
            if 'color' in data.keys():
                grid_image, _ = create_grid_image(pts, res_item, data['gt_pose'], data['color'], self.cfg.pose_mode, self.cfg.o2c_pose)
            else:
                grid_image, _ = create_grid_image(pts, res_item, data['gt_pose'], None, self.cfg.pose_mode, self.cfg.o2c_pose)
            self.writer.add_image(f'{mode}/vis_{sampler_item}', grid_image, self.clock.epoch)          
    
    def collect_metric(self, pred_pose, gt_pose, cat_ids,):
        rot_error, trans_error, symtr_error = get_errors(
            pred_pose.type_as(gt_pose),
            gt_pose,
            class_ids=cat_ids,
        )
        rot_error = {
            'mean': np.mean(rot_error),
            'median': np.median(rot_error),
            'item': rot_error,
        }
        trans_error = {
            'mean': np.mean(trans_error),
            'median': np.median(trans_error),
            'item': trans_error,
        }
        error = {'rot_error': rot_error,
                 'trans_error': trans_error}
        return error