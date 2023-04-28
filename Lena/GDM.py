from config import get_arguments
from SGANs.manipulate import *
from SGANs.training import *
from SGANs.imresize import imresize
import SGANs.functions as functions
from sklearn.manifold import MDS
from multiprocessing.pool import ThreadPool
from cal_entropy import cal_entropy
import matplotlib.pyplot as plt
import torch
import numpy as np

parser = get_arguments()
parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
parser.add_argument('--input_name', help='input image name', default='lena_multi.png')
parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', default='random_samples')
parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1)
parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)
opt = parser.parse_args()
opt = functions.post_config(opt)
Gs = []
Zs = []
reals = []
NoiseAmp = []
dir2save = functions.generate_dir2save(opt)
real = functions.read_image(opt)
real = functions.denorm(real)
functions.adjust_scales2image(real, opt)
Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)

#%%
torch.pi = torch.acos(torch.zeros(1)).item() * 2

r_opt = 0

loss_opt = 0

num_iter = 100

num_realization = 100

num_r = 11

location = np.array([[205,325],[200,290],[190,250],[200,230],[212,204],[220,195],[240,225],[270,240],[203,182],
                      [182,200],[167,207],[156,217]])

images_curr = []

#%%
real_image = np.array(np.squeeze(real.to(torch.device('cpu'))))
real_image = np.where(real_image > 0.67, 2, (np.where(real_image < 0.33,0,1)))

Zs_GDM_0, I_curr_0 = SGANs_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)

Zs_GDM_1, I_curr_1 = SGANs_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)

plt.figure()
plt.imshow(real_image,cmap='gray')  
plt.scatter(location[:,1],location[:,0],c='r')      

image_0 = functions.convert_image_np(I_curr_0.detach())
image_0 = np.where(image_0 > 0.67, 2, (np.where(image_0 < 0.33,0,1)))

plt.figure()
plt.imshow(image_0,cmap='gray')
plt.scatter(location[:,1],location[:,0],c='r')      

image_1 = functions.convert_image_np(I_curr_1.detach())
image_1 = np.where(image_1 > 0.67, 2, (np.where(image_1 < 0.33,0,1)))

plt.figure()
plt.imshow(image_1,cmap='gray')
plt.scatter(location[:,1],location[:,0],c='r')      

#%%

while len(images_curr) < num_realization:
    
    print('The total number is %d' %len(images_curr))
    
    Zs_GDM_0, _ = SGANs_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)

    Zs_GDM_1, _ = SGANs_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)
    
    index_image = 0
    
    loss_opt = 0  # have to check this

    for iter in range(num_iter):
        
        index = 0
           
        print('The iteration number is %d' %iter)
    
        for r in range(num_r):
            
            theta = torch.as_tensor(r / ((num_r-1)*2))
            
            _, I_opt = SGANs_generate_GDM(Gs, Zs, Zs_GDM_0, Zs_GDM_1, theta, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)
        
            
            image = np.array(np.squeeze(functions.convert_image_np(I_opt.detach())))       
            image = np.where(image > 0.67, 2, (np.where(image < 0.33,0,1)))
                      
            loss_curr = np.sum(image[location[:,0],location[:,1]])
            
            if loss_curr > loss_opt:
                index = 1
                loss_opt = loss_curr
                r_opt = r
                print('The loss is %0.4f' %loss_opt)
                
            if r == num_r - 1 and index == 1:    
            
                theta = torch.as_tensor(r_opt / ((num_r-1)*2))
                
                _, I_opt = SGANs_generate_GDM(Gs, Zs, Zs_GDM_0, Zs_GDM_1, theta, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)
                
                image_opt = functions.convert_image_np(I_opt.detach())
                image_opt = np.where(image_opt > 0.67, 2, (np.where(image_opt < 0.33,0,1)))
                
                Zs_GDM_0, _ = SGANs_generate_GDM(Gs, Zs, Zs_GDM_0, Zs_GDM_1, theta, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)
                
            if loss_curr == 2*len(location):
                images_curr.append(image)
                index_image = 1
                break
        
        if index_image == 1:
            break
        
        Zs_GDM_1, _ = SGANs_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)
          

#%%
plt.figure()
plt.imshow(images_curr[0],'gray')

#%%
pool = ThreadPool()

cal_temp = np.asarray(images_curr).transpose(1,2,0).tolist()

cal_res = pool.map(cal_entropy, [[cal_temp[i]] for i in range(np.asarray(cal_temp).shape[0])])

pool.close()

pool.join()

cal_ent = np.squeeze(np.asarray(cal_res))
#%%
plt.figure()
plt.imshow(cal_ent,'gray_r')
plt.scatter(location[:,1],location[:,0],c='r')    

#%%
random_real = []
for i in range(num_realization):
    _, I_curr = SGANs_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)
    image = functions.convert_image_np(I_curr.detach())
    image = np.where(image > 0.67, 2, (np.where(image < 0.33,0,1)))
    random_real.append(image)  

pool = ThreadPool()

sim_temp = np.asarray(random_real).transpose(1,2,0).tolist()

sim_res = pool.map(cal_entropy, [[sim_temp[i]] for i in range(np.asarray(sim_temp).shape[0])])

pool.close()

pool.join()

sim_ent = np.squeeze(np.asarray(sim_res))
#%%
plt.figure()
plt.imshow(sim_ent,'gray_r')
plt.scatter(location[:,1],location[:,0],c='r')           
#%%
model_data = np.zeros((1+len(random_real),random_real[0].shape[0]*random_real[0].shape[1]))        
model_data[0,:] = real_image.ravel()
model_data[1:,:] = np.asarray(random_real).reshape(len(random_real),random_real[0].shape[0]*random_real[0].shape[1])

embedding = MDS(n_components=2)
model_MDS = embedding.fit_transform(model_data)

plt.figure()
plt.scatter(model_MDS[0,0],model_MDS[0,1],c='r')
plt.scatter(model_MDS[1:,0],model_MDS[1:,1],c='k')

        
        
        











