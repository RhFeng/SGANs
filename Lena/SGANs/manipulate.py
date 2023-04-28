from __future__ import print_function
from SGANs.imresize import imresize
import SGANs.functions as functions
import torch.nn as nn
import torch
from config import get_arguments

def SGANs_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=1):
    
    Zs_GDM = []
    
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    images_cur = []
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            if n == 0:
                z_curr = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_curr = z_curr.expand(1,opt.nc_z,z_curr.shape[2],z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)

            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)
                if opt.mode != "SR":
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                    I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
                else:
                    I_prev = m(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp*(z_curr)+I_prev
            I_curr = G(z_in.detach(),I_prev)


            images_cur.append(I_curr)
        n+=1
        
        Zs_GDM.append(z_curr)
    return Zs_GDM, I_curr

def SGANs_generate_GDM(Gs,Zs,Zs_GDM_0,Zs_GDM_1,theta,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=1):
    
    Zs_GDM = []
     
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    images_cur = []
    for G,Z_opt,z_in_0,z_in_1,noise_amp in zip(Gs,Zs,Zs_GDM_0,Zs_GDM_1,NoiseAmp):
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            if n == 0:
                z_curr = z_in_1.detach() * torch.sin(theta * torch.pi) + z_in_0.detach() * torch.cos(theta * torch.pi)
            else:
                z_curr = z_in_1.detach() * torch.sin(theta * torch.pi) + z_in_0.detach() * torch.cos(theta * torch.pi)

            if images_prev == []:
                I_prev = m(in_s)
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)
                if opt.mode != "SR":
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                    I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
                else:
                    I_prev = m(I_prev)

            z_in = noise_amp*(z_curr)+I_prev
            I_curr = G(z_in.detach(),I_prev)


            images_cur.append(I_curr)
        n+=1
        
        Zs_GDM.append(z_curr)
    return Zs_GDM, I_curr

