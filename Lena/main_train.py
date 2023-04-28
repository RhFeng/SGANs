from config import get_arguments
from SGANs.manipulate import *
from SGANs.training import *
import SGANs.functions as functions
import numpy as np
import os

parser = get_arguments()
parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
parser.add_argument('--input_name', help='input image name', default='lena_multi.png')
parser.add_argument('--mode', help='task to be done', default='train')

alpha = np.array([10]) #0.1 0.5 1 5 10

#%%
# Training
for i in range(len(alpha)):
    
    parameters = {'alpha': alpha[i]}

    parser.set_defaults(**parameters)
    
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    0
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    
    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        print('total scale number is %d' % (opt.stop_scale))
        train(opt, Gs, Zs, reals, NoiseAmp)






        