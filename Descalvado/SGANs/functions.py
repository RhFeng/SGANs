import torch
import numpy as np
import torch.nn as nn
import math
from skimage import io as img
from SGANs.imresize import imresize
import os
import random

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def convert_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:,:])
        inp = inp.numpy().transpose((0,1,2))
    inp = np.clip(inp,0,1)
    return inp

def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:,:])
        inp = inp.numpy().transpose((0,1,2))

    inp = np.clip(inp,0,1)
    return inp[100,:,:]

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), round(size[3]/scale), device=device)
        noise = upsampling(noise,size[1], size[2], size[3])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], size[3], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], size[3], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], size[3], device=device)
    return noise

def upsampling(im,sx,sy,sz):
    m = nn.Upsample(size=[round(sx),round(sy),round(sz)],mode='trilinear',align_corners=True)
    return m(im)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def read_image(opt):
    x = np.load('%s/%s' % (opt.input_dir,opt.input_name))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:,:]
    return x

def np2torch(x,opt):
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255        
    else:
        x = x[:,:,:,None,None]
        x = x.transpose(4, 3, 0, 1, 2)/255
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    x = norm(x)
    return x

def torch2uint8(x):
    x = x[0,:,:,:,:]
    x = x.permute((1,2,3,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def save_networks(netG,netD,z,opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

def adjust_scales2image(real_,opt):
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3], real_.shape[4])), 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3], real_.shape[4]])]) / max([real_.shape[2], real_.shape[3], real_.shape[4]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3], real_.shape[4]]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3],real.shape[4])),1/(opt.stop_scale))
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3],real_.shape[4]])]) / max([real_.shape[2], real_.shape[3], real_.shape[4]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def creat_reals_pyramid(real,reals,opt):
    real = real[:,0:3,:,:,:]
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize(real,scale,opt)
        reals.append(curr_real)
    return reals

def load_trained_pyramid(opt, mode_='train'):
    mode = opt.mode
    opt.mode = 'train'
    dir = generate_dir2save(opt)
    if(os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir)
        Zs = torch.load('%s/Zs.pth' % dir)
        reals = torch.load('%s/reals.pth' % dir)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == 'train'):
        dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%.1f' % (opt.input_name[:-4], opt.scale_factor_init,opt.alpha)
    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/scale_factor=%f,alpha=%.1f/gen_start_scale=%d' % (opt.out, opt.input_name[:-4],opt.scale_factor_init,opt.alpha, opt.gen_start_scale)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
    return dir2save

def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt



