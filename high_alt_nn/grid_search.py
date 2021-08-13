#--------------------------------------------------#
# grid_search.py                                   #
# Author: Eric Ham                                 #
# Description: Contains functions for grid search  #
# over models.                                     #
#--------------------------------------------------#

import numpy as np
import itertools
from train_lunar_net import train_model
# consraint 2
def constraint2(a, b, g, margin):
    err = a*(b**2)*(g**2) - 2
    if np.abs(err) < margin:
        return True
    return False


# generate list of parameter combinations
# subject to the constraints:
#   a >=1, b >= 1, g >= 1
#   a*(b^2)*(g^2) ~ 2 (within some margin)
def get_grid_params(margin, precision):
    # note on range choices:
    # set margin at too large value (ex. 1), min is when all = 1. 
    # one at a time, set others to 1, determine its max. 
    # in this case, get 3, root 3, root 3 respectively
    # add precision, so get upper limit including precision (not on pb or pg bc
    # 2 > root 3

    # get all possible values for parameters (note consraint 1 is applied here
    pa = np.arange(1, 3 + precision, precision).round(2)
    pb = np.arange(1, 2, precision).round(2)
    pg = np.arange(1,2, precision).round(2)
    #import pdb; pdb.set_trace()    
    # get all possible combinations of the above parameters
    pos_combs = list(itertools.product(pa, pb, pg))
    
    # rule out combinations based on the co
    filtered_combs = []
    for val in pos_combs:
        if constraint2(val[0], val[1], val[2], margin):
            filtered_combs.append(val)
    
    return filtered_combs

def run_search(combs_list, model_ps, dg_ps):
    for comb in comb_list:
        a = comb[0]
        b = comb[1]
        g = comb[2]
        #model = set_model(model_params)
        #model.set_hyper_ps(a,b,g)
        
        

if __name__ == '__main__':
    combs = get_grid_params(0.5, 0.05)
    #import pdb; pdb.set_trace()
