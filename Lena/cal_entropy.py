from scipy.stats import entropy
import numpy as np

def cal_entropy(temp):
    num_facies = 3
    temp = np.squeeze(np.asarray(temp))
    sim_ent = np.zeros((temp.shape[0],1))
    for i in range(temp.shape[0]):
        prop = np.zeros((num_facies,1))
        for facies in range(num_facies):
            prop[facies] = np.count_nonzero(temp[i,:] == facies)
  
        if np.sum(prop) != temp.shape[1]:
            print(np.sum(prop))
            print('check the data!')
        sim_ent[i,0] = entropy(prop / temp.shape[1],base = num_facies)
    return sim_ent