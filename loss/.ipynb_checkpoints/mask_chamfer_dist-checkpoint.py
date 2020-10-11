import torch
import torch.nn as nn

INF = 1000000

def chamfer_loss(A,B, mask1, mask2, ps=91, clip=10000):
    A=A.permute(0,2,1)
    B=B.permute(0,2,1)
    r=torch.sum(A*A,dim=2)
    r=r.unsqueeze(-1)
    r1=torch.sum(B*B,dim=2)
    r1=r1.unsqueeze(-1)
    
    t= r.repeat(1,1,ps) - 2*torch.bmm(A,B.permute(0,2,1)) + r1.permute(0, 2, 1).repeat(1,ps,1)
    
    d1,_=t.min(dim=2)
    d1 = d1*mask1
    mask1=((1.0-(d1>=clip)).float())*mask1
    d1 =  mask1*d1 
    d2,_=t.min(dim=1)
    d2=d2*mask2
    
    mask2=((1.0-(d2>=clip)).float())*mask2
    d2=(mask2*d2)
    ls=(d1+d2)/2
    
    return ls.mean(), mask1, mask2