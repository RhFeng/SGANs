from __future__ import print_function
from __future__ import print_function
from SGANs.imresize import imresize
import SGANs.functions as functions
import torch.nn as nn
import torch
from config import get_arguments
import torch.nn.functional as F

def SGANs_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,scale_z=1,n=0,gen_start_scale=0,num_samples=1):
   
    Zs_GDM = []
    
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    images_cur = []
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = (int(pad1),int(pad1),int(pad1),int(pad1),int(pad1),int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h
        nzz = (Z_opt.shape[4]-pad1*2)*scale_z

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            if n == 0:
                z_curr = functions.generate_noise([1,nzx,nzy,nzz], device=opt.device)
                z_curr = z_curr.expand(1,opt.nc_z,z_curr.shape[2],z_curr.shape[3],z_curr.shape[4])
                z_curr = F.pad(z_curr,m,'constant',0)
            else:
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy,nzz], device=opt.device)
                z_curr = F.pad(z_curr,m,'constant',0)

            if images_prev == []:
                I_prev = F.pad(in_s,m,'constant',0)
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)
                if opt.mode != "SR":
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3]), 0:round(scale_z * reals[n].shape[4])]
                    I_prev = F.pad(I_prev,m,'constant',0)
                    I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3],0:z_curr.shape[4]]
                    I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3],z_curr.shape[4])
                else:
                    I_prev = F.pad(I_prev,m,'constant',0)

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp*(z_curr)+I_prev
            I_curr = G(z_in.detach(),I_prev)


            images_cur.append(I_curr)
        n+=1
        
        Zs_GDM.append(z_curr)
    return I_curr
