from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from .networks import LocNetReg2D, LocNetRegAVD, MLP
from utils import transform_to_global_2D, transform_to_global_AVD
import math

def get_M_net_inputs_labels(occupied_points, unoccupited_points):
    """
    get global coord (occupied and unoccupied) and corresponding labels
    """
    n_pos = occupied_points.shape[1]
    inputs = torch.cat((occupied_points, unoccupited_points), 1)
    bs, N, _ = inputs.shape

    gt = torch.zeros([bs, N, 1], device=occupied_points.device)
    gt.requires_grad_(False)
    gt[:, :n_pos, :] = 1
    return inputs, gt


def sample_unoccupied_point(local_point_cloud, n_samples, center):
    """
    sample unoccupied points along rays in local point cloud
    local_point_cloud: <BxLxk>
    n_samples: number of samples on each ray
    center: location of sensor <Bx1xk>
    """
    bs, L, k = local_point_cloud.shape
    
    center = center.expand(-1,L,-1) # <BxLxk>
    unoccupied = torch.zeros(bs, L * n_samples, k,
                             device=local_point_cloud.device)
    for idx in range(1, n_samples + 1):
        fac = torch.rand(1).item()
        unoccupied[:, (idx - 1) * L:idx * L, :] = center + (local_point_cloud-center) * fac
    return unoccupied

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        print(pe.size())
        
    def forward(self, index):
        return self.pe[index,:]
    '''
    def forward(self, x):
        x = torch.cat((x,self.pe[:x.size(0), :]), dim=1)
        return self.dropout(x)
    '''
class DeepMapping2D(nn.Module):
    def __init__(self, loss_fn, n_obs=256, z_size=64, n_samples=19, dim=[2, 64, 512, 512, 256, 128, 1]):
        super(DeepMapping2D, self).__init__()
        self.n_obs = n_obs
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.loc_net = LocNetReg2D(n_points=n_obs,z_size=z_size, out_dims=3)
        self.occup_net = MLP(dim)
        self.masks = None
        self.epoch = 0

    def forward(self, vec, obs_local,valid_points,sensor_pose, masks,epoch):
        # obs_local: <BxLx2>
        # vec : <Bx1xz>
        # sensor_pose: init pose <Bx1x3>
        self.masks = masks
        self.obs_local = deepcopy(obs_local)
        vec = vec.expand(-1,obs_local.size(1),-1)
        obs_local_enhanced = torch.cat((obs_local,vec),dim=2)
        self.valid_points = valid_points
        self.epoch = epoch
        
        self.pose_est = self.loc_net(obs_local_enhanced)

        self.obs_global_est = transform_to_global_2D(
            self.pose_est, self.obs_local)

        if self.training:
            sensor_center = sensor_pose[:,:,:2]
            self.unoccupied_local = sample_unoccupied_point(
                self.obs_local, self.n_samples,sensor_center)
            self.unoccupied_global = transform_to_global_2D(
                self.pose_est, self.unoccupied_local)

            inputs, self.gt = get_M_net_inputs_labels(
                self.obs_global_est, self.unoccupied_global)
            self.occp_prob = self.occup_net(inputs)
            return self.compute_loss()

    def compute_loss(self):
        valid_unoccupied_points = self.valid_points.repeat(1, self.n_samples)
        bce_weight = torch.cat(
            (self.valid_points, valid_unoccupied_points), 1).float()
        # <Bx(n+1)Lx1> same as occp_prob and gt
        bce_weight = bce_weight.unsqueeze(-1)

        if self.loss_fn.__name__ == 'bce_ch':
            return  self.loss_fn(self.occp_prob, self.gt, self.obs_global_est,
                                self.valid_points, bce_weight, seq=2, gamma=0.1)  # BCE_CH
        elif self.loss_fn.__name__ == 'bce':
            return self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE
            
        elif self.loss_fn.__name__ == 'bce_chmk':
            return self.loss_fn(self.occp_prob, self.gt, self.obs_global_est,
                                self.valid_points, bce_weight, seq=2, gamma=0.1,masks=self.masks,epoch = self.epoch) 
        
class DeepMapping_AVD(nn.Module):
    #def __init__(self, loss_fn, n_samples=35, dim=[3, 256, 256, 256, 256, 256, 256, 1]):
    def __init__(self, loss_fn, z_size=64, n_samples=35, dim=[3, 64, 512, 512, 256, 128, 1]):
        super(DeepMapping_AVD, self).__init__()
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.loc_net = LocNetRegAVD(z_size=z_size,out_dims=3) # <x,z,theta> y=0
        self.occup_net = MLP(dim)
        self.masks = None
        self.epoch = 0
    
    def forward(self,vec,obs_local,valid_points,sensor_pose,masks,epoch):
        # obs_local: <BxHxWx3> 
        # valid_points: <BxHxW>
        self.epoch = epoch
        self.masks = deepcopy(masks)
        self.obs_local = deepcopy(obs_local)

        vec = vec.unsqueeze(1).expand(-1,obs_local.size(1),obs_local.size(2),-1)
        obs_local_enhanced = torch.cat((obs_local,vec),dim=3)
        
        self.valid_points = valid_points
        self.pose_est = self.loc_net(obs_local_enhanced)

        bs = obs_local.shape[0]
        self.obs_local = self.obs_local.view(bs,-1,3)
        self.valid_points = self.valid_points.view(bs,-1)
        
        self.obs_global_est = transform_to_global_AVD(
            self.pose_est, self.obs_local)

        if self.training:
            sensor_center = sensor_pose
            self.unoccupied_local = sample_unoccupied_point(
                self.obs_local, self.n_samples,sensor_center)
            self.unoccupied_global = transform_to_global_AVD(
                self.pose_est, self.unoccupied_local)

            inputs, self.gt = get_M_net_inputs_labels(
                self.obs_global_est, self.unoccupied_global)
            self.occp_prob = self.occup_net(inputs)
            return self.compute_loss()
            

    def compute_loss(self):
        valid_unoccupied_points = self.valid_points.repeat(1, self.n_samples)
        bce_weight = torch.cat(
            (self.valid_points, valid_unoccupied_points), 1).float()
        # <Bx(n+1)Lx1> same as occp_prob and gt
        bce_weight = bce_weight.unsqueeze(-1)

        if self.loss_fn.__name__ == 'bce_ch':
            return self.loss_fn(self.occp_prob, self.gt, self.obs_global_est,
                                self.valid_points, bce_weight, seq=2, gamma=0.9)  # BCE_CH
             
        elif self.loss_fn.__name__ == 'bce':
            return self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE

        elif self.loss_fn.__name__ == 'bce_chmk':
            return self.loss_fn(self.occp_prob, self.gt, self.obs_global_est,
                                self.valid_points, bce_weight, seq=2, gamma=0.9, masks=self.masks,epoch=self.epoch)

        


