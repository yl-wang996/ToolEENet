import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.genpose_utils import get_pose_dim

'''
    scorenet.py: for generating the pose candidates 
'''


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()  # ? (bs, in_features) @ (in_features, out_features) -> (bs, out_features)
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x




class PoseScoreNet(nn.Module):
    def __init__(self, marginal_prob_func, pose_mode='rot_matrix', regression_head='RT'):
        super(PoseScoreNet, self).__init__()
        self.regression_head = regression_head
        self.activation = nn.ReLU(True)
        pose_dim = get_pose_dim(pose_mode)
        ''' encode pose '''
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            self.activation,
            nn.Linear(256, 256),
            self.activation,
        )
        
        ''' encode t '''
        self.t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            # self.act, # M4D26 update
            nn.Linear(128, 128),
            self.activation,
        )

        ''' fusion tail '''
        if self.regression_head == 'Rx_Ry_and_T':
            ''' rotation_x_axis regress head '''
            self.fusion_tail_rot_x = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                self.activation,
                zero_module(nn.Linear(256, 3)),
            )
            self.fusion_tail_rot_y = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                self.activation,
                zero_module(nn.Linear(256, 3)),
            )
            
            ''' translation regression head '''
            self.fusion_tail_trans = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                self.activation,
                zero_module(nn.Linear(256, 3)),
            )
        elif self.regression_head == 'Rx_Ry_and_T_and_Symtr':
            ''' rotation_x_axis regress head '''
            self.fusion_tail_rot_x = nn.Sequential(
                nn.Linear(128 + 256 + 1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                self.activation,
                zero_module(nn.Linear(256, 3)),
            )
            self.fusion_tail_rot_y = nn.Sequential(
                nn.Linear(128 + 256 + 1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                self.activation,
                zero_module(nn.Linear(256, 3)),
            )
            
            ''' translation regression head '''
            self.fusion_tail_trans = nn.Sequential(
                nn.Linear(128 + 256 + 1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                self.activation,
                zero_module(nn.Linear(256, 3)),
            )
            '''estimate the symmetric properties of the pose'''
            self.fusion_tail_symtr = nn.Sequential(
                nn.Linear(128 + 256 + 1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                self.activation,
                zero_module(nn.Linear(256, 3)),
            )
        else:
            raise NotImplementedError
            
        self.marginal_prob_func = marginal_prob_func


    def forward(self, data):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, c]
                'pose_sample': [bs, pose_dim]
                't': [bs, 1]
            }
        '''
        
        pts_feat = data['pts_feat']
        sampled_pose = data['sampled_pose']
        t = data['t']
        
        # pts = pts.permute(0, 2, 1) # -> (bs, 3, 1024)
        # pts_feat = self.pts_encoder(pts)
        
        t_feat = self.t_encoder(t.squeeze(1))
        pose_feat = self.pose_encoder(sampled_pose)

        total_feat = torch.cat([pts_feat, t_feat, pose_feat], dim=-1)
        _, std = self.marginal_prob_func(total_feat, t)
        
        if self.regression_head == 'Rx_Ry_and_T':
            rot_x = self.fusion_tail_rot_x(total_feat)
            rot_y = self.fusion_tail_rot_y(total_feat)
            trans = self.fusion_tail_trans(total_feat)
            out_score = torch.cat([rot_x, rot_y, trans], dim=-1) / (std+1e-7) # normalisation
        elif self.regression_head == 'Rx_Ry_and_T_and_Symtr':
            rot_x = self.fusion_tail_rot_x(total_feat)
            rot_y = self.fusion_tail_rot_y(total_feat)
            trans = self.fusion_tail_trans(total_feat)
            symtr = self.fusion_tail_symtr(total_feat)
            out_score = torch.cat([rot_x, rot_y, trans, symtr], dim=-1) / (std + 1e-7)  # normalisation
        else:
            raise NotImplementedError
        return out_score


# class PoseDecoderNet(nn.Module):
#     def __init__(self, marginal_prob_func, sigma_data=1.4148, pose_mode='quat_wxyz', regression_head='RT'):
#         """_summary_
#
#         Args:
#             marginal_prob_func (func): marginal_prob_func of score network
#             pose_mode (str, optional): the type of pose representation from {'quat_wxyz', 'quat_xyzw', 'rot_matrix', 'euler_xyz'}. Defaults to 'quat_wxyz'.
#             regression_head (str, optional): _description_. Defaults to 'RT'.
#
#         Raises:
#             NotImplementedError: _description_
#         """
#         super(PoseDecoderNet, self).__init__()
#         self.sigma_data = sigma_data
#         self.regression_head = regression_head
#         self.act = nn.ReLU(True)
#         pose_dim = get_pose_dim(pose_mode)
#
#         ''' encode pose '''
#         self.pose_encoder = nn.Sequential(
#             nn.Linear(pose_dim, 256),
#             self.act,
#             nn.Linear(256, 256),
#             self.act,
#         )
#
#         ''' encode sigma(t) '''
#         self.sigma_encoder = nn.Sequential(
#             PositionalEmbedding(num_channels=128),
#             nn.Linear(128, 128),
#             self.act,
#         )
#
#         ''' fusion tail '''
#         init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0) # init the final output layer's weights to zeros
#
#         if self.regression_head == 'Rx_Ry_and_T':
#             if pose_mode != 'rot_matrix':
#                 raise NotImplementedError
#             ''' rotation_x_axis regress head '''
#             self.fusion_tail_rot_x = nn.Sequential(
#                 nn.Linear(128+256+1024, 256),
#                 # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 self.act,
#                 Linear(256, 3, **init_zero),
#             )
#             self.fusion_tail_rot_y = nn.Sequential(
#                 nn.Linear(128+256+1024, 256),
#                 # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 self.act,
#                 Linear(256, 3, **init_zero),
#             )
#
#             ''' tranalation regress head '''
#             self.fusion_tail_trans = nn.Sequential(
#                 nn.Linear(128+256+1024, 256),
#                 # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 self.act,
#                 Linear(256, 3, **init_zero),
#             )
#         elif self.regression_head == 'Rx_Ry_and_T_and_Symtr':
#             if pose_mode != 'rot_matrix':
#                 raise NotImplementedError
#             ''' rotation_x_axis regress head '''
#             self.fusion_tail_rot_x = nn.Sequential(
#                 nn.Linear(128 + 256 + 1024, 256),
#                 # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 self.act,
#                 Linear(256, 3, **init_zero),
#             )
#             self.fusion_tail_rot_y = nn.Sequential(
#                 nn.Linear(128 + 256 + 1024, 256),
#                 # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 self.act,
#                 Linear(256, 3, **init_zero),
#             )
#
#             ''' tranalation regress head '''
#             self.fusion_tail_trans = nn.Sequential(
#                 nn.Linear(128 + 256 + 1024, 256),
#                 # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 self.act,
#                 Linear(256, 3, **init_zero),
#             )
#
#             '''estimate the symmetric properties of the pose'''
#             self.fusion_tail_symtr = nn.Sequential(
#                 nn.Linear(128 + 256 + 1024, 256),
#                 # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 self.act,
#                 Linear(256, 3, **init_zero),
#             )
#
#         else:
#             raise NotImplementedError
#
#         self.marginal_prob_func = marginal_prob_func
#
#
#     def forward(self, data):
#         '''
#         Args:
#             data, dict {
#                 'pts_feat': [bs, c]
#                 'pose_sample': [bs, pose_dim]
#                 't': [bs, 1]
#             }
#         '''
#
#         pts_feat = data['pts_feat']
#         sampled_pose = data['sampled_pose']
#         t = data['t']
#         _, sigma_t = self.marginal_prob_func(None, t) # \sigma(t) = t in EDM
#
#         # determine scaling functions
#         # EDM
#         # c_skip = self.sigma_data ** 2 / (sigma_t ** 2 + self.sigma_data ** 2)
#         # c_out = self.sigma_data * t / torch.sqrt(sigma_t ** 2 + self.sigma_data ** 2)
#         # c_in = 1 / torch.sqrt(sigma_t ** 2 + self.sigma_data ** 2)
#         # c_noise = torch.log(sigma_t) / 4
#         # VE
#         c_skip = 1
#         c_out = sigma_t
#         c_in = 1
#         c_noise = torch.log(sigma_t / 2)
#
#         # comp total feat
#         sampled_pose_rescale = sampled_pose * c_in
#         pose_feat = self.pose_encoder(sampled_pose_rescale)
#         sigma_feat = self.sigma_encoder(c_noise.squeeze(1))
#         total_feat = torch.cat([pts_feat, sigma_feat, pose_feat], dim=-1)
#
#         if self.regression_head == 'RT':
#             nn_output = self.fusion_tail(total_feat)
#         elif self.regression_head == 'R_and_T':
#             rot = self.fusion_tail_rot(total_feat)
#             trans = self.fusion_tail_trans(total_feat)
#             nn_output = torch.cat([rot, trans], dim=-1)
#         elif self.regression_head == 'Rx_Ry_and_T':
#             rot_x = self.fusion_tail_rot_x(total_feat)
#             rot_y = self.fusion_tail_rot_y(total_feat)
#             trans = self.fusion_tail_trans(total_feat)
#             nn_output = torch.cat([rot_x, rot_y, trans], dim=-1)
#         elif self.regression_head == 'Rx_Ry_and_T_and_Symtr':
#             rot_x = self.fusion_tail_rot_x(total_feat)
#             rot_y = self.fusion_tail_rot_y(total_feat)
#             trans = self.fusion_tail_trans(total_feat)
#             symtr = self.fusion_tail_symtr(total_feat)
#             nn_output = torch.cat([rot_x, rot_y, trans, symtr], dim=-1)
#         else:
#             print(f"regression_head ({self.regression_head} is not supported")
#             raise NotImplementedError
#
#         denoised_output = c_skip * sampled_pose + c_out * nn_output
#         return denoised_output

