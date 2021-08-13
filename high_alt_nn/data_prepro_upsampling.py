#-------------------------------------------------------#
# Code Relies on data prepro functionality.             #
# Uses upsampling instead of downsampling to simplify   #
# xyz dataset construction              #
#-------------------------------------------------------#

import os
import numpy as np
import math
from copy import deepcopy
import itertools
import resource
from sys import stdout, getrefcount
import gc

from data_prepro import get_out_xy, im_samps_from_points, img_load, im_train_val_split_with_skips, to_image, standardize_xy
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
from upsampling_main import full_procedure, just_bilinear, just_diamond_square

def mem(desc=''):
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stdout.write('[Mem] %s\t %dKb\n' % (desc, usage))


# test point to img
def test_pts_to_img(t_x, t_y, new_t_pts_list, t_samps):
    # debugging (need to verify the points correspond with the images) 
    # samples are same, so check that the points are in the same location in the image. 
    # by outputting the image sample, looking at overall image and point. 
    # just do for t (v is exactly the same). 2 points on each altitude. samp 0, 100
    #breakpoint()
    save_img(os.path.join(os.getcwd(), '1xt_0.png'), t_x[0])
    save_img(os.path.join(os.getcwd(), '1xt_100.png'), t_x[100])
    print(t_y[0])
    print(t_y[100])    
    
    save_img(os.path.join(os.getcwd(), '2xt_0.png'), t_samps[0][0])
    save_img(os.path.join(os.getcwd(), '2xt_100.png'), t_samps[0][100])
    print(new_t_pts_list[0][0])
    print(new_t_pts_list[0][100])
        
    save_img(os.path.join(os.getcwd(), '4xt_0.png'), t_samps[1][0])
    save_img(os.path.join(os.getcwd(), '4xt_100.png'), t_samps[1][100])
    print(new_t_pts_list[1][0])
    print(new_t_pts_list[1][100])

    save_img(os.path.join(os.getcwd(), '6xt_0.png'), t_samps[2][0])
    save_img(os.path.join(os.getcwd(), '6xt_100.png'), t_samps[2][100])
    print(new_t_pts_list[2][0])
    print(new_t_pts_list[2][100])

def test_up_ims(num_ups, up_ims):
    s_dir = os.getcwd()
    for i in range(num_ups):
        print(up_ims[i].shape)
        out_f = os.path.join(s_dir, str(2*i) + 'x.png')
        save_img(out_f, np.expand_dims(up_ims[i],-1))

# given original points, make an array of points for each altitude. they are the same x, y but
# an altitude (z) component is added. highest z is least upsampled (original). 
def add_alts(pts, num_alts):
    alts = []
    for i in range(num_alts):
            alts.append(np.hstack((pts, (num_alts-i)*np.ones((len(pts), 1)))))
    
    return alts

# swap xy positions for val and train sets for a number of altitudes. since the v/t split is fixed in a given
# altitude, this means you will have to accept your overall v/t split will be shifted by some margin
def swap_vt_alts(train_set, val_set, val_p, margin):
    #breakpoint()
    #new_train, new_val = [], []
    # determine number of swaps
    n_alts = len(train_set)
    n_swaps = math.ceil(n_alts*(margin/(1-2*val_p))) # round up so at least 1 swap
    # get swaps evenly spaced
    swap_inds = []
    if n_swaps == 1:
        # handle this case separately since i % 1 == 0 for all i. 
        swap_inds.append(math.floor(n_alts/2)) # n_alts = 3 gives 1, 4 gives 2 (in middle when possible)
        #temp = train_set[swap_idx]
        #train_set[swap_idx] = val_set[swap_idx]
        #val_set[swap_idx] = temp
    else: 
        # note: expect swaps to be fewer than non-swaps (since don't want big change in v/t split)
        # so we want swaps to be centered to allow interpolation demonstration
        # since 0 % x == 0 for all x, we skip this case. 
        for i in range(n_alts):
            if (i+1) % n_swaps == 0:
                #train_set.append(val_set[i])
                #val_set.append(train_set[i])
                swap_inds.append(i)
            #else:
            #    train_set.append(train_set[i]) 
            #    val_set.append(val_set[i])

    #return train_set, val_set
    return swap_inds

class integrated_data_wrapper:
    #@profile
    def __init__(self,im_a, w, l, skip_amt, val_amt, vbs, margin, num_ups, pred_type, up_f, *up_args):
        t_x,t_y,v_x,v_y = im_train_val_split_with_skips(im_a, w, l,skip_amt,val_amt, vbs, get_out_xy)
        # upsample the image
        # can upsample 2x, 4x, 6x, 8x, 10x etc. 
        up_ims = [up_f(im_a, 2*i, *up_args) for i in range(1,num_ups+1)]
        up_ims = list(map(lambda x: np.expand_dims(x, -1), up_ims))
        # debugging test
        #test_up_ims(num_ups, up_ims)
           
        # getting points for upsampled images (og x, y have moved in these images, so in order
        # to get the images, we needed to shift our x,y positions)
        # add altitude here. 
        # note: num_ups + 1 - i (num_ups + 1 = og). then down to 1. 
        # ex. num_ups = 3. 4 alts. 4 = og, 3 = 2x, 2 = 4x, 1 = 6x
        # we make the largest upsampling the "highest" altitude (largest number)
        # new_t_pts_list (and for v) is size (num_ups, num_points, 3 dims)
        z_t = np.expand_dims(np.ones(t_y.shape[0]), -1) # altitude 
        t_samps = [np.hstack((t_y*2*i,z_t*(num_ups+1-i))) for i in range(1, num_ups+1)]
 
        z_v =  np.expand_dims(np.ones(v_y.shape[0]), -1)
        v_samps = [np.hstack((v_y*2*i, z_v*(num_ups+1-i))) for i in range(1, num_ups+1)]

        # reminder that im_samps_from_points takes a list of points as input 
        # make sure giving up_ims not the original
        t_samps = [im_samps_from_points(t_samps[i], w, l, up_ims[i]) for i in range(num_ups)]
        v_samps = [im_samps_from_points(v_samps[i], w, l, up_ims[i]) for i in range(num_ups)]
         
        # debugging test
        #test_pts_to_img(t_x, t_y, new_t_pts_list, t_samps)
        
        # assemble the dataset
        # NOTE: even though the xy position is different in the upsampled images, the ground truth x,y is the same. so you use the xy from the original image NOT the new one. The new one is ONLY generated to get the image samples. 
        
        t_samps.insert(0, t_x)
        v_samps.insert(0, v_x)

        if pred_type == 'z': 
            t_y = add_alts(t_y, num_ups+1)   
            v_y = add_alts(v_y, num_ups+1)
        else:
            t_y = [t_y for i in range(num_ups+1)]
            v_y = [v_y for i in range(num_ups+1)]

        swap_inds = swap_vt_alts(t_samps, v_samps, val_amt, margin)
        for i in swap_inds:
            temp = t_samps[i]
            t_samps[i] = v_samps[i]
            v_samps[i] = temp
            
            out_temp = t_y[i]
            t_y[i] = v_y[i]
            v_y[i] = out_temp

        
    
        # remove outer array so that we have big lists of samples. 
        t_samps = np.asarray(list(itertools.chain.from_iterable(t_samps)))        
        v_samps = np.asarray(list(itertools.chain.from_iterable(v_samps)))        
        t_y = np.asarray(list(itertools.chain.from_iterable(t_y)))
        v_y = np.asarray(list(itertools.chain.from_iterable(v_y)))
        
        if pred_type == 'xy':
            print('xy')
            t_y, v_y, w_med, w_iqr, l_med, l_iqr = standardize_xy(im_a, t_y,v_y)
            self.w_med = w_med
            self.w_iqr = w_iqr
            self.l_med = l_med
            self.l_iqr = l_iqr
            
            self.image = im_a
            self.x_train = t_samps
            self.y_train = t_y
            self.x_val = v_samps 
            self.y_val = v_y
            print(type(t_samps))
            print(type(t_y))
            print(type(v_samps))
            print(type(v_y))

        elif pred_type == 'z':
            # add altitudes
            print('z')
            ts_f = []
            vs_f = []
            for val in t_y:
                ts_f.extend([val[-1] for val in t_y])
                vs_f.extend([val[-1] for val in v_y])
            z_vals = list(range(1, num_ups+2)) # get altitudes
            z_med = np.median(z_vals)
            zq75, zq25 = np.percentile(z_vals, [75 ,25])
            z_iqr = zq75 - zq25
            self.z_iqr = z_iqr
            self.z_med = z_med
            self.x_train = np.asarray([ex for ex in t_samps])
            self.y_train = np.array([((ex-z_med)/z_iqr) for ex in ts_f])
            self.x_val = np.array([ex for ex in v_samps])
            self.y_val = np.array([((ex-z_med)/z_iqr) for ex in vs_f])
            

        ''' Setting up Wrapper (do later)
        if axes == 'all':
            self.x_train = np.asarray([ex[0] for ex in train_pts])
            self.y_train = np.asarray([ex[1] for ex in train_pts])
            self.x_val = np.asarray([ex[0] for ex in val_pts])
            self.y_val = np.asarray([ex[1] for ex in val_pts])
        elif axes  == 'z':
            z_vals = list(range(1, len(final_points)+1))
            z_med = np.median(z_vals)
            zq75, zq25 = np.percentile(z_vals, [75 ,25])
            z_iqr = zq75 - zq25
            self.z_iqr = z_iqr
            self.z_med = z_med
            self.x_train = np.asarray([ex[0] for ex in train_pts])
            self.y_train = np.array([((ex[1][-1]-z_med)/z_iqr) for ex in train_pts])
            self.x_val = np.array([ex[0] for ex in val_pts])
            self.y_val = np.array([((ex[1][-1]-z_med)/z_iqr) for ex in val_pts])
            
        #import pdb; pdb.set_trace()
        print('Train Set Size: ',len(self.x_train), 'Val Set Size: ',len(self.x_val))
        '''
if __name__ == '__main__':
    img_path = '../images/quarry_images/up_test2/'
    ch = 600
    cl = 600
    imgs = ['up2_20', 'up2_30', 'up2_60', 'up2_120']
    im_a = None
    for f in imgs:
        im_a = img_load(img_path+f+'.png')
        y,x = im_a.shape
        sx = x//2-(ch//2)
        sy = y//2-(cl//2)
        im_a = im_a[sy:sy+ch, sx:sx+cl]
        im = Image.fromarray(im_a)
        im.save(img_path+"cropped"+f.split('_')[1]+'.png')
    w = 21
    l = 21
    skip_amt = 0
    val_amt = 0.2
    vbs = 1
    samp_size = 21
    num_ups = 1
    margin = 0.1
    pred_type = 'xy'
    ds = integrated_data_wrapper(im_a, w, l, skip_amt, val_amt,vbs,margin, num_ups,pred_type, just_bilinear)
