import os
import sys

import numpy as np
import torch
from scipy import integrate

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.genpose_utils import get_pose_dim
from utils.misc import normalize_rotation


def global_prior_likelihood(z, sigma_max):
    """The likelihood of a Gaussian distribution with mean zero and 
        standard deviation sigma."""
    # z: [bs, pose_dim]
    shape = z.shape
    N = np.prod(shape[1:]) # pose_dim
    return -N / 2. * torch.log(2*np.pi*sigma_max**2) - torch.sum(z**2, dim=-1) / (2 * sigma_max**2)


def cond_ode_likelihood(
        score_model,
        data,
        prior,
        sde_coeff,
        marginal_prob_fn,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        num_steps=None,
        pose_mode='rot_matrix',
        init_x=None,
    ):
    
    pose_dim = get_pose_dim(pose_mode)
    batch_size = data['pts'].shape[0]
    epsilon = prior((batch_size, pose_dim)).to(device)
    init_x = data['sampled_pose'].clone().cpu().numpy() if init_x is None else init_x
    shape = init_x.shape
    init_logp = np.zeros((shape[0],)) # [bs]
    init_inp = np.concatenate([init_x.reshape(-1), init_logp], axis=0)
    
    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        return score.cpu().numpy().reshape((-1,))
    
    def divergence_eval(data, epsilon):
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        # save ckpt of sampled_pose
        origin_sampled_pose = data['sampled_pose'].clone()
        with torch.enable_grad():
            # make sampled_pose differentiable
            data['sampled_pose'].requires_grad_(True)
            score = score_model(data)
            score_energy = torch.sum(score * epsilon) # [, ]
            grad_score_energy = torch.autograd.grad(score_energy, data['sampled_pose'])[0] # [bs, pose_dim]
        # reset sampled_pose
        data['sampled_pose'] = origin_sampled_pose
        return torch.sum(grad_score_energy * epsilon, dim=-1) # [bs, 1]
    
    def divergence_eval_wrapper(data):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad(): 
            # Compute likelihood.
            div = divergence_eval(data, epsilon) # [bs, 1]
        return div.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def ode_func(t, inp):        
        """The ODE function for use by the ODE solver."""
        # split x, logp from inp
        x = inp[:-shape[0]]
        logp = inp[-shape[0]:] # haha, actually we do not need use logp here
        # calc x-grad
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        x_grad = drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
        # calc logp-grad
        logp_grad = drift - 0.5 * (diffusion**2) * divergence_eval_wrapper(data)
        # concat curr grad
        return  np.concatenate([x_grad, logp_grad], axis=0)
     
    # Run the black-box ODE solver
    res = integrate.solve_ivp(ode_func, (eps, 1.0), init_inp, rtol=rtol, atol=atol, method='RK45')
    zp = torch.tensor(res.y[:, -1], device=device) # [bs * (pose_dim + 1)]
    z = zp[:-shape[0]].reshape(shape) # [bs, pose_dim]
    delta_logp = zp[-shape[0]:].reshape(shape[0]) # [bs,] logp
    _, sigma_max = marginal_prob_fn(None, torch.tensor(1.).to(device)) # we assume T = 1 
    prior_logp = global_prior_likelihood(z, sigma_max)
    log_likelihoods = (prior_logp + delta_logp) / np.log(2) # negative log-likelihoods (nlls)
    return z, log_likelihoods

def cond_ode_sampler(
        score_model,
        data,
        prior,
        sde_coeff,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        T=1.0,
        num_steps=None,
        pose_mode='rot_matrix',
        denoise=True,
        init_x=None,
    ):
    assert pose_mode in ['rot_matrix', 'rot_matrix_symtr'], f"the rotation mode {pose_mode} is not supported!"
    pose_dim = get_pose_dim(pose_mode)
    batch_size=data['pts'].shape[0]
    init_x = prior((batch_size, pose_dim), T=T).to(device) if init_x is None else init_x + prior((batch_size, pose_dim), T=T).to(device)
    shape = init_x.shape
    
    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        return score.cpu().numpy().reshape((-1,))
    
    def ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        return drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
  
    # Run the black-box ODE solver
    t_eval = None
    if num_steps is not None:
        # num_steps, from T -> eps
        t_eval = np.linspace(T, eps, num_steps)
    res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, pose_dim) # [num_steps, bs, pose_dim]
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape) # [bs, pose_dim]
    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['sampled_pose'] = x.float()
        data['t'] = vec_eps
        grad = score_model(data)
        drift = drift - diffusion**2*grad       # R-SDE
        mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        x = mean_x
    
    num_steps = xs.shape[0]
    xs = xs.reshape(batch_size*num_steps, -1)
    if pose_mode == 'rot_matrix_symtr':
        xs[:, :6] = normalize_rotation(xs[:, :6], pose_mode)
        xs = xs.reshape(num_steps, batch_size, -1)
        xs[:, :, 6:9] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
        x[:, :6] = normalize_rotation(x[:, :6], pose_mode)
        x[:, 6:-3] += data['pts_center']
    else:
        xs[:, :-3] = normalize_rotation(xs[:, :-3], pose_mode)
        xs = xs.reshape(num_steps, batch_size, -1)
        xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
        x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
        x[:, -3:] += data['pts_center']
    return xs.permute(1, 0, 2), x