import set_path
import os
import argparse
import functools
print = functools.partial(print,flush=True)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as tini

import utils
import loss
from models import DeepMapping2D, PositionalEncoding
from dataset_loader import SimulatedPointCloud

torch.backends.cudnn.deterministic = True
torch.manual_seed(4242)
os.environ['CUDA_VISIBLE_DEVICES'] = '8'
parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='test',help='experiment name')
parser.add_argument('-e','--n_epochs',type=int,default=1000,help='number of epochs')
parser.add_argument('-b','--batch_size',type=int,default=32,help='batch_size')
parser.add_argument('-l','--loss',type=str,default='bce_chmk',help='loss function')
parser.add_argument('-n','--n_samples',type=int,default=19,help='number of sampled unoccupied points along rays')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('-d','--data_dir',type=str,default='../data/2D/',help='dataset path')
parser.add_argument('-m','--model', type=str, default=None,help='pretrained model name')
parser.add_argument('-i','--init', type=str, default=None,help='init pose')
parser.add_argument('--log_interval',type=int,default=10,help='logging interval of saving results')
parser.add_argument('-s','--num_lat',type=int,default=64,help='size of embedding')

opt = parser.parse_args()
print(opt)
checkpoint_dir = os.path.join('../results/2D',opt.loss,opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
utils.save_opt(checkpoint_dir,opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('loading dataset')
if opt.init is not None:
    init_pose_np = np.load(opt.init)
    init_pose = torch.from_numpy(init_pose_np)
else:
    init_pose = None
    
dataset = SimulatedPointCloud(opt.data_dir,init_pose)
loader = DataLoader(dataset,batch_size=opt.batch_size,shuffle=False)

#pe = PositionalEncoding(d_model=opt.num_lat,max_len=len(dataset))
latent_vecs = []
mask_vecs_pair = []

for i in range(len(dataset)):
    #vec = tini.xavier_normal_(torch.ones((1,opt.num_lat))).to(device)
    vec = (torch.ones(1,opt.num_lat).normal_(0, 0.5).to(device))
    vec = torch.nn.Parameter(vec) #True
    latent_vecs.append(vec)
    
    mask_vec_back = torch.ones(dataset.point_clouds.size()[1]).to(device)
    mask_vec_front = torch.ones(dataset.point_clouds.size()[1]).to(device)
    mask_vecs_pair.append([mask_vec_back,mask_vec_front]) 
    
loss_fn = eval('loss.'+opt.loss)

print('creating model')
model = DeepMapping2D(loss_fn=loss_fn,n_obs=dataset.n_obs,z_size= opt.num_lat, n_samples=opt.n_samples).to(device)
optimizer = optim.Adam(model.parameters(),lr=opt.lr)

if opt.model is not None:
    utils.load_checkpoint(opt.model,model,optimizer)

print('start training')
for epoch in range(opt.n_epochs):

    training_loss= 0.0
    model.train()

    for index,(indeces,obs_batch,valid_pt,pose_batch) in enumerate(loader):
        obs_batch = obs_batch.to(device)
        valid_pt = valid_pt.to(device)
        pose_batch = pose_batch.to(device)
        l_vec = torch.stack([latent_vecs[i] for i in indeces.numpy()])
        #p_vec = pe(indeces).to(device)
        #full_vec = torch.cat((l_vec,p_vec),dim=2)
        full_vec = l_vec
        loss,new_masks = model(full_vec,obs_batch,valid_pt,pose_batch,mask_vecs_pair,epoch)
        mask_vecs_pair = new_masks
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
    
    training_loss_epoch = training_loss/len(loader)

    if (epoch+1) % opt.log_interval == 0:
        print('[{}/{}], training loss: {:.4f}'.format(
            epoch+1,opt.n_epochs,training_loss_epoch))

        obs_global_est_np = []
        pose_est_np = []
        with torch.no_grad():
            model.eval()
            for index,(indeces,obs_batch,valid_pt,pose_batch) in enumerate(loader):
                obs_batch = obs_batch.to(device)
                valid_pt = valid_pt.to(device)
                pose_batch = pose_batch.to(device)
                l_vec = torch.stack([latent_vecs[i] for i in indeces.numpy()])
                # p_vec = pe(indeces).to(device)
                # l_vec = torch.cat((l_vec,p_vec),dim=2)
                full_vec = l_vec 
                model(full_vec,obs_batch,valid_pt,pose_batch,mask_vecs_pair,epoch)

                obs_global_est_np.append(model.obs_global_est.cpu().detach().numpy())
                pose_est_np.append(model.pose_est.cpu().detach().numpy())
            
            pose_est_np = np.concatenate(pose_est_np)
            if init_pose is not None:
                pose_est_np = utils.cat_pose_2D(init_pose_np,pose_est_np)

            save_name = os.path.join(checkpoint_dir,opt.loss+'_model_best.pth')
            utils.save_checkpoint(save_name,model,optimizer)

            obs_global_est_np = np.concatenate(obs_global_est_np)
            kwargs = {'e':epoch+1}
            
            valid_pt_np = dataset.valid_points.cpu().detach().numpy()
            print(dataset.valid_points.size())
            masks_npy = np.array([mask_vecs_pair[0][1].cpu().detach().numpy(),mask_vecs_pair[1][0].cpu().detach().numpy()])
            print(masks_npy.shape)
            utils.plot_global_point_cloud(obs_global_est_np,pose_est_np,valid_pt_np,checkpoint_dir,**kwargs)

            #save_name = os.path.join(checkpoint_dir,'obs_global_est.npy')
            #np.save(save_name,obs_global_est_np)

            save_name = os.path.join(checkpoint_dir,opt.loss+'_pose_est.npy')
            np.save(save_name,pose_est_np)

