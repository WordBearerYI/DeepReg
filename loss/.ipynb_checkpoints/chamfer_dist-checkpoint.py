import torch
import torch.nn as nn
import copy
INF = 1000000


class ChamfersDistance(nn.Module):
    '''
    Extensively search to compute the Chamfersdistance. 
    '''
    
    def forward(self, input1, input2, valid1=None, valid2=None):
        # input1, input2: BxNxK, BxMxK, K = 3
        B, N, K = input1.shape
        _, M, _ = input2.shape
        if valid1 is not None:
            # ignore invalid points
            valid1 = valid1.type(torch.float32)
            valid2 = valid2.type(torch.float32)

            invalid1 = 1 - valid1.unsqueeze(-1).expand(-1, -1, K)
            invalid2 = 1 - valid2.unsqueeze(-1).expand(-1, -1, K)

            input1 = input1 + invalid1 * INF * torch.ones_like(input1)
            input2 = input2 + invalid2 * INF * torch.ones_like(input2)

        # Repeat (x,y,z) M times in a row
        input11 = input1.unsqueeze(2)           # BxNx1xK
        input11 = input11.expand(B, N, M, K)    # BxNxMxK
        # Repeat (x,y,z) N times in a column
        input22 = input2.unsqueeze(1)           # Bx1xMxK
        input22 = input22.expand(B, N, M, K)    # BxNxMxK
        # compute the distance matrix
        D = input11 - input22                   # BxNxMxK
        D = torch.norm(D, p=2, dim=3)         # BxNxM

        dist0, _ = torch.min(D, dim=1)        # BxM
        dist1, _ = torch.min(D, dim=2)        # BxN

        if valid1 is not None:
            dist0 = torch.sum(dist0 * valid2, 1) / torch.sum(valid2, 1)
            dist1 = torch.sum(dist1 * valid1, 1) / torch.sum(valid1, 1)
        else:
            dist0 = torch.mean(dist0, 1)
            dist1 = torch.mean(dist1, 1)

        loss = dist0 + dist1  # B
        loss = torch.mean(loss)                             # 1
        return loss


def registration_loss(obs, valid_obs=None):
    """
    Registration consistency
    obs: <BxLx2> a set of obs frame in the same coordinate system
    select of frame as reference (ref_id) and the rest as target
    compute chamfer distance between each target frame and reference

    valid_obs: <BxL> indics of valid points in obs
    """
    criternion = ChamfersDistance()
    bs = obs.shape[0]
    ref_id = 0
    ref_map = obs[ref_id, :, :].unsqueeze(0).expand(bs - 1, -1, -1)
    valid_ref = valid_obs[ref_id, :].unsqueeze(0).expand(bs - 1, -1)

    tgt_list = list(range(bs))
    tgt_list.pop(ref_id)
    tgt_map = obs[tgt_list, :, :]
    valid_tgt = valid_obs[tgt_list, :]
    loss = criternion(ref_map, tgt_map, valid_ref, valid_tgt)
    return loss

def update_mask(cur_obs,tar_obs,cur_valid,ps,clip):
    n_p = cur_obs.size(1)
    #print(cur_valid[0,:].size(),cur_obs.size(),tar_obs.size(),n_p)
    
    cur_obs *= cur_valid[0,:].unsqueeze(-1).expand(cur_obs.size(0),n_p)
    tar_obs *= cur_valid[1,:].unsqueeze(-1).expand(tar_obs.size(0),n_p)
    dist_mat = torch.cdist(cur_obs,tar_obs)
    dist_min_from_cur,_ = dist_mat.min(dim=0)
    dist_min_from_tar,_ = dist_mat.min(dim=1)
    points_remain_cur = (~(dist_min_from_cur>=clip)).float()
    points_remain_tar = (~(dist_min_from_cur>=clip)).float()
    return points_remain_cur,points_remain_tar

def chamfer_loss_dynamic_mask(obs,valid_obs,masks=None,seq=2,ps=91,clip=2,epoch=0):
    #only support a pair for now
    #obs: B*N*3, valid_obs: [B]*N
    assert masks is not None
    bs = obs.shape[0]
    total_step = bs - seq + 1
    loss = 0.
    new_masks = copy.deepcopy(masks)
    # 存储每一对chamfer 用以找出异常（卡住）
    if epoch>2000:
        clip = max(200.0/(epoch-2000+1),0.01)
    loss_lst = []
    for step in range(total_step):
        current_obs = obs[step:step+seq]
        current_valid = valid_obs[step:step + seq].long()
        current_valid[0,:] *= masks[step][1].long()
        current_valid[1,:] *= masks[step+1][0].long()
        
        #if step==0:
        #    print(step,current_mask,current_valid)
        
        current_loss = registration_loss(current_obs, current_valid)
        loss_lst.append(round(current_loss.item(),6))
        new_masks[step] = update_mask(current_obs[0],current_obs[1],current_valid,ps,clip)
        loss = loss + current_loss
    
    loss = loss / total_step
    return loss, new_masks


def chamfer_loss(obs, valid_obs=None, seq=2):
    bs = obs.shape[0]
    
    total_step = bs - seq + 1
    loss = 0.
    for step in range(total_step):
        current_obs = obs[step:step + seq]
        current_valid_obs = valid_obs[step:step + seq]

        current_loss = registration_loss(current_obs, current_valid_obs)
        loss = loss + current_loss

    loss = loss / total_step
    return loss, None

    
'''
def update_mask(cur_obs,tar_obs,cur_valid,ps,clip):
    dist_mat = torch.cdist(cur_obs,tar_obs)
    dist_min_from_cur,_ = dist_mat.min(dim=0)
    #print(cur_valid.size(),cur_valid)
    #print(dist_min_from_cur,dist_min_from_cur.size(),int(0.8*(dist_min_from_cur.size(0))))
    #dist_min_from_cur = dist_min_from_cur*cur_valid[0,:]
    #points_to_remain = (~(dist_min_from_cur>=clip)).float()
    #remain_values,remain_indeces = torch.topk(dist_min_from_cur,int(0.75*(dist_min_from_cur.size(0))),largest=False)
    #clip = remain_values.max()  
    points_to_remain = (~(dist_min_from_cur>=clip)).float()
    #print('clip',clip, remain_values, remain_indeces)
    #print('updated',points_to_remain)
    return points_to_remain
'''
