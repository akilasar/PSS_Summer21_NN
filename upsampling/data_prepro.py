# ------------------------------------------------------------------------------#
# data preprocessing code
# takes an image as input, breaks into subimages of a given width and height
# with each image differing from the next by a shift of one pixel to the
# right. same for up and down. then take samples in between (by defined number
# of skips) for validation set. the label for each image is the center pixel's
# x and y coordinate. 
# Author: Eric Ham
# Last Edited: 7/14/20
# ------------------------------------------------------------------------------#
   
#from PIL import Image
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
import scipy.stats as st
random_value = 42
np.random.seed(random_value)
#from scipy

# load an image from the given path handle greyscale 
def img_load(image_path):
    im = load_img(image_path, color_mode='grayscale')
    im_a = img_to_array(im)
    im_mean = np.mean(im_a)
    im_std = np.std(im_a)
    im_a -= im_mean
    im_a /= im_std
    #print(im_a.shape)
    return im_a

# convert array to image, and store it in Image_Tests directory
def to_image(im_a,name):
    save_dir = '/Users/emham/Desktop/PSSWork/LunarNet/Image_Tests/'
    file_name = 'im_test_' + name + '.png'
    #img = array_to_img(im_a)
    #print(im_a.shape)    
    save_img(save_dir + file_name, im_a)

# get true value when its the center
def get_out_xy(i,j,w,l,im_a, new_samp):
    x_cent = j + math.floor(l/2.0)
    y_cent = i + math.floor(w/2.0)
    samp_x_cent = math.floor(l/2.0)
    samp_y_cent = math.floor(w/2.0)
    # any is True if any values are 1. else --> False
    assert(not any(new_samp[samp_y_cent, samp_x_cent] - im_a[y_cent,x_cent]))
    
    return float(y_cent) , float(x_cent)


def get_out_rt(i,j,w,l, pad1, pad2):
    x_cent = j + math.floor(l/2.0)
    y_cent = i + math.floor(w/2.0)
    r = np.sqrt(y_cent**2 + x_cent**2)
    t = np.arctan(y_cent/x_cent)
    return float(r),float(t)

def standardize_xy(im_a, y_t,y_v):
    #import pdb; pdb.set_trace()
    # reference point is top left corner (within the image)
    w = im_a.shape[0]
    w_vals = np.arange(0,w)
    w_med = np.median(w_vals)
    wq75, wq25 = np.percentile(w_vals, [75 ,25])
    w_iqr = wq75 - wq25
    y_t[:,0] -= float(w_med)
    y_t[:,0] /= w_iqr
    y_v[:,0] -= w_med
    y_v[:,0] /= w_iqr

    l = im_a.shape[1]
    l_vals = np.arange(0,l)
    l_med = np.median(l_vals)
    lq75, lq25 = np.percentile(l_vals, [75 ,25])
    l_iqr = lq75 - lq25
    y_t[:,1] -= l_med
    y_t[:,1] /= l_iqr
    y_v[:,1] -= l_med
    y_v[:,1] /= l_iqr

    return y_t, y_v, w_med, w_iqr, l_med, l_iqr

def standardize_rt(y_t, y_v):
    # reference point is top left corner
    # r min is 0 in theory
    # r max is the diagonal length (in theory)

    # t min is 0 in theory (right next to top left)
    # t max is pi/2 in theory (right below top left)
    
    # just use all the data
    all_rs = []
    all_rs.extend(list(y_t[:,0]))
    all_rs.extend(list(y_v[:,0]))
    r_mean = np.mean(all_rs)
    r_std = np.std(all_rs)

    y_t[:,0] -= r_mean
    y_t[:,0] /= r_std
    y_v[:,0] -= r_mean
    y_v[:,0] /= r_std

    all_ts = []
    all_ts.extend(list(y_t[:,1]))
    all_ts.extend(list(y_v[:,1]))
    t_mean = np.mean(all_ts)
    t_std = np.std(all_ts)

    y_t[:,1] -= t_mean
    y_t[:,1] /= t_std
    y_v[:,1] -= t_mean
    y_v[:,1] /= t_std

    return y_t, y_v, r_mean, r_std, t_mean, t_std



# returns the max overlap you will get if you keep shifting right by 1 and down
# by 1 for the given starting width and overlap
def max_shift_overlap(w, overlap,l):
    maximizing_shift = math.floor((w - overlap)/2)
    if overlap + maximizing_shift >= l:
        out = (w - maximizing_shift)*(l)
        
    else: 
        out = (w - maximizing_shift)*(overlap + maximizing_shift)

    return 

# plot boxes around training/val images in different colors. can use centers,
# width/length values to draw boxes. 
def plot_data_set(train_y, val_y, w, l, im_w, im_l):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    samp_w = l # their width is our length (x)
    samp_h = w # their height is our width (y)
    for ex in train_y:
        # convert to lower left corner
        # our ex[0] is row, so y here. ex[1] is col, so x here
        bottom_left = ((ex[1] - (l-1)/2),im_w -(ex[0] - (w-1)/2))

        r = Rectangle(bottom_left, samp_w, -samp_h, color='blue', fill=False,lw=1, ls = '--')
        ax.add_patch(r)
    
    for ex in val_y:
        # convert to lower left corner
        # our ex[0] is row, so y here. ex[1] is col, so x here
        bottom_left = ((ex[1] - (l-1)/2) ,im_w - (ex[0] - (w-1)/2))

        r = Rectangle(bottom_left, samp_w, -samp_h, color='orange',fill=False,lw=1,ls = '--')
        ax.add_patch(r)
    
    # test
    #im_w is y values here. (im.shape[0])
    #im_l is x values here. (im.shape[1])
    plt.xlim([0,im_l])
    plt.ylim([0,im_w])
    plt.show()

# split image array into all subimages of given size (in pixels) (l, w must be
# odd for center pixel)
# note: l corresponds to x-axis (horizontal), w corresponds to y-axis (vertical)
def split_image_array(im_a, w, l):
    num_row_samps = im_a.shape[1] - l + 1 #number of samples in a single row
    num_col_samps = im_a.shape[0] - w + 1 #number of samples in a single column
    x_vals = []
    y_vals = []
    for i in range(num_col_samps): 
        for j in range(num_row_samps):
            new_samp = im_a[i:i+w,j:j+l,:]
            #print(new_samp.shape)
            assert(new_samp.shape == (w,l,1))
            x_vals.append(new_samp)
            x_cent = j + math.floor(l/2.0)
            y_cent = i + math.floor(w/2.0)
            y_vals.append([y_cent, x_cent]) #[row, col]
            samp_x_cent = math.floor(l/2.0)
            samp_y_cent = math.floor(w/2.0)
            #print(samp_y_cent, samp_x_cent)
            # any is True if any values are 1. else --> False
            assert(not any(new_samp[samp_y_cent, samp_x_cent] - im_a[y_cent,
                x_cent]))
            
    
    #make sure all samples are of size w, l, 3
    assert(len(x_vals) == len(y_vals))
    assert(len(x_vals) == num_row_samps*num_col_samps)
    print('Total Num Samples: ',len(x_vals))    
    return np.array(x_vals), np.array(y_vals)

# split samples into training and validation set
def im_train_val_split(x_vals, y_vals, val_amt):
    
    skip_amt = int(1.0/(val_amt)) # for example, if val amount is 0.2 --> get 4, so VTTTTVTTTT
    val_inds = np.arange(0, x_vals.shape[0], skip_amt) #skip amount = 1-->VTVT 2 --> VTTVTT
    all_inds = np.arange(x_vals.shape[0])
    train_inds = np.delete(all_inds, val_inds)
    #print(val_inds)
    x_train = x_vals[train_inds]
    y_train = y_vals[train_inds]
    x_val = x_vals[val_inds]
    y_val = y_vals[val_inds]
        
    #print(y_val.shape)
    #print(val_amt)
    #print(y_vals.shape)
    assert((y_val.shape[0] -  y_vals.shape[0]*val_amt) <= 1.0)
    #print(val_inds[0:20])
    #print(train_inds[0:20])
    #print(val_inds[-20:])
    #print(train_inds[-20:])
    return x_train, y_train, x_val, y_val

def im_train_val_split_with_skips(im_a, w, l, skip_amt, val_amt, vbs,
        output_func):
    num_row_samps = im_a.shape[1] - l + 1 #number of samples in a single row(ie. num cols)
    num_col_samps = im_a.shape[0] - w + 1 #number of samples in a single column (ie. num rows)
    val_x_vals = []
    val_y_vals = []
    train_x_vals = []
    train_y_vals = []
    #paper equations to find num_val_blocks per row. 
    #nvbs_pr=math.floor(val_amt*(num_row_samps*num_col_samps+skip_amt)*(1/((vbs+skip_amt*2*val_amt)*num_row_samps)))# num valblocks per row
    #import pdb; pdb.set_trace()
    nvbs_pr = math.ceil(val_amt*num_col_samps*(num_row_samps + skip_amt) *
            (1/(num_col_samps*(vbs + val_amt*skip_amt*2))))
    skip_pr = (2*nvbs_pr - 1)*skip_amt
    #train_pr = num_row_samps - (skip_pr + nvbs_pr*vbs)
    train_pr = math.ceil(nvbs_pr*vbs*(1-val_amt)/(val_amt))
    #print(train_pr, nvbs_pr*vbs)
    tbs = math.ceil(train_pr/nvbs_pr)
    # see if need to add extra to last train block
    last_train = train_pr % nvbs_pr
    print('extra train',last_train)
    # row values
    vb_pr_count = 0 # val block per row count
    in_val_block_count = 0

    skip_count = 0
    
    prev_i = 0
   
    # start in val
    state = 'val'
    prev_state = 'start'

    tb_pr_count = 0
    in_train_block_count = 0
    last_train_count = 0
    
    #column values
    c_state = 'val'
    in_v_c_count = 0
    skip_c_amt = 0

    test = 0
    test_2 = 0
    test_3 = 0
    test_4 = 0
    total_v_count = 0
    #import pdb; pdb.set_trace()
    #num_col_samps = 9
    overlap = l - (skip_amt + 1)
    
    s = (w - overlap)
    if s > l - overlap:
        s = math.ceil((w/l)*(l-overlap))
    elif s <= 0:
        s = 1 # don't do this: + skip_amt. it would result in equalizing
        #regardless of skip_amt. point of skip_amt is to space more. taken into
        #account above with overlap
    s_count = 0
    # in practice, you want to skip s - 1 rows. ex. w = 10, o = 3, s = 7. but
    # want to start next at 8 --> skip 2,3,4,5,6,7 (6).
    s = s - 1 # can combine this above, but leaving it here for understanding. 
    shift_type = 'reg'
    last_shift = 'skip'
    #ok_area_of_overlap = w*(l-(skip_amt + 1)) # skip_amt = 0 --> overlap is (l -1)*w
    #print('s val: ', s)
    for i in range(0,num_col_samps):
        total_v_count +=1
        if s_count > 0 and s_count <= s:
            s_count +=1 
            continue
        elif s_count == 0:
            s_count+=1
        # s > 0 so s_count > 0 is redundant
        elif s_count > s:
            s_count = 1
            t = last_shift
            last_shift = shift_type
            shift_type = t

        in_skip_count = -1
        #in_skip_count = 0
        for j in range(num_row_samps):
            #print(state, j)  
            
            if shift_type == 'skip' and in_skip_count <s:
                in_skip_count+=1
                continue

            # at each new row, return to val state and reset all counts
            if prev_i != i:

                #print(nvbs_pr, vb_pr_count)
                
                #print(vb_pr_count, nvbs_pr)
                #assert(vb_pr_count == nvbs_pr)
                state = 'val'
                in_val_block_count = 0
                skip_count = 0
                vb_pr_count = 0
                tb_pr_count = 0
                in_train_block_count = 0

            prev_i = i

            if state == 'val':
                new_samp = im_a[i:i+w,j:j+l,:]
                assert(new_samp.shape == (w,l,1))
                val_x_vals.append(new_samp)
                
                #x_cent = j + math.floor(l/2.0)
                #y_cent = i + math.floor(w/2.0)
                #val_y_vals.append([y_cent, x_cent]) #[row, col]
                #samp_x_cent = math.floor(l/2.0)
                #samp_y_cent = math.floor(w/2.0)
                # any is True if any values are 1. else --> False
                #assert(not any(new_samp[samp_y_cent, samp_x_cent] - im_a[y_cent,x_cent]))
                y_1, y_2 = output_func(i,j,w,l,im_a, new_samp)
                val_y_vals.append([y_1, y_2]) #[row, col]
                
                in_val_block_count +=1
                if in_val_block_count == vbs:
                    in_val_block_count = 0
                    if skip_amt != 0:
                        state = 'skip'
                    else:
                        state = 'train'
                    prev_state = 'val'
                    vb_pr_count +=1
                
            elif state == 'skip':
                skip_count+=1
                if skip_count == skip_amt and prev_state == 'val':
                    state = 'train'
                    prev_state = 'skip'
                    skip_count = 0
                elif skip_count == skip_amt and prev_state == 'train':
                    state = 'val'
                    skip_count = 0
                    prev_state = 'skip'
                else:
                    state = 'skip'
            
            elif state == 'train':
                test+=1
                # normal train block operation 
                #import pdb; pdb.set_trace()
                new_samp = im_a[i:i+w,j:j+l,:]
                assert(new_samp.shape == (w,l,1))
                train_x_vals.append(new_samp)
                #x_cent = j + math.floor(l/2.0)
                #y_cent = i + math.floor(w/2.0)
                #train_y_vals.append([y_cent, x_cent]) #[row, col]
                #samp_x_cent = math.floor(l/2.0)
                #samp_y_cent = math.floor(w/2.0)
                # any is True if any values are 1. else --> False
                #assert(not any(new_samp[samp_y_cent, samp_x_cent] - im_a[y_cent,x_cent]))
                y_1, y_2 = output_func(i,j,w,l,im_a, new_samp)
                train_y_vals.append([y_1, y_2]) #[row, col]
                
                if in_train_block_count != tbs:
                    in_train_block_count +=1
                if in_train_block_count == tbs and (tb_pr_count+1) == nvbs_pr:
                    test_3 +=1
                    #import pdb; pdb.set_trace() 
                    # since we have completed a block, increment the train
                    # block counter
                    # if counting this block would make it the last block, then
                    # handle that case
                    #print(i,j)
                    last_train_count+=1
                    # compare count with expected remainder and exit state
                    # if complete
                    if last_train_count > last_train:
                        test_4 +=1
                        #import pdb; pdb.set_trace()
                        last_train_count = 0
                        tb_pr_count = 0 
                        in_train_block_count = 0
                        state = 'skip'
                        if skip_amt != 0:
                            state = 'skip'
                        else:
                            state = 'val'

                        prev_state = 'train'

                # otherwise, go back to skip state
                elif in_train_block_count == tbs and (tb_pr_count+1) != nvbs_pr:
                        test_2+=1
                        tb_pr_count += 1
                        in_train_block_count = 0
                        if skip_amt != 0:
                            state = 'skip'
                        else:
                            state = 'val'
                        prev_state = 'train'
    
    train_x_vals, train_y_vals, val_x_vals, val_y_vals = np.asarray(train_x_vals), np.asarray(train_y_vals), np.asarray(val_x_vals),np.asarray(val_y_vals)
    print('train in: ',train_x_vals.shape)
    print('train out: ',train_y_vals.shape)
    print('val in: ',val_x_vals.shape)
    print('val out: ',val_y_vals.shape)
    #print(test, test_2, test_3, test_4, total_v_count)
    #print(last_train) 
    #assert((y_val.shape[0] -  y_vals.shape[0]*val_amt) <= 1.0)
    #print(val_inds[0:20])
    #print(train_inds[0:20])
    #print(val_inds[-20:])
    #print(train_inds[-20:])
    return train_x_vals, train_y_vals, val_x_vals, val_y_vals

# filt_size: filter size (filter dim is (filt_size, filt_size))
# gauss_amt: is the amount of the gaussian N(0,I) in the filter. default is 3
# because -3 to 3 covers most of gaussian we want. beyond it is ~ 0.
def gauss_filt(filt_size, gauss_amt = 3):
    x = np.linspace(-gauss_amt, gauss_amt, filt_size + 1) # need extra value for within pixel below
    filt1d = np.diff(st.norm.cdf(x)) # compute probability mass within each pixel
    # multiplying two gaussians gives joint distribution. outer product to get matrix 
    # (this makes sense since 2d discrete joint distribution can be seen as a
    # matrix)
    filt2d = np.outer(filt1d, filt1d)
    return filt2d/filt2d.sum() # return normalized filter

def plot_filt(filt):
    plt.imshow(filt, interpolation='none')
    plt.show()

def strided_conv(filt_type,filt_size, stride, im_a, all_pts):
    if filt_type == 'avg':
        filt = np.ones((filt_size, filt_size))
        filt /= (filt_size*filt_size)
    elif filt_type == 'gaussian':
        filt = gauss_filt(filt_size)
    
    #distance to samp center from sides:
    # distance to samp center from top/bottom:
    # center vals will still be in the image (border may not be though)
    # need to keep track of new values (with stride and edges (d_to_c))
    prev_points = [] # original center values. we want to keep original x,y,
    # but will now have new coordinates for inputs
    #distance to filter center
    #d_to_fc = filt_size //2
    new_y = 0
    new_x = 0
    new_points = [] #the values of the points in prev_center_vals in the
    
    # new image
    prev_w = im_a.shape[0]
    prev_l = im_a.shape[1]
    new_w = prev_w - 2*(filt_size//2)
    new_l = prev_l - 2*(filt_size//2)
    if new_w % stride != 0:
        # start with included one, so if odd, will always be at least 1 kept
        # point
        new_w = math.ceil(new_w/stride)
    else:
        new_w /= stride
    
    if new_l % stride != 0:
        new_l = math.ceil(new_l/stride)
    else:
        new_l /= stride

    new_w = int(new_w)
    new_l = int(new_l)
    #print(new_w, new_l)
    new_im = np.zeros((new_w, new_l))
    #import pdb; pdb.set_trace()
    #print(im_a.shape)
    #print(new_im.shape)
    #import pdb; pdb.set_trace()
    assert_pass_count = 0
    for i in range(0, im_a.shape[0]-filter_size + 1, stride):
        new_x = 0
        for j in range(0, im_a.shape[1] -filter_size+ 1, stride):
            center = [i + filt_size//2, j + filt_size//2]
            #import pdb; pdb.set_trace()
            #print(assert_pass_count, new_y,new_w,new_x,new_l)
            #if assert_pass_count == 164:
            #    import pdb; pdb.set_trace()
            #assert(tuple(center) in all_pts)
            #if tuple(center) not in set(all_pts):
            #    continue
            assert_pass_count+=1 
            # get the location of the point in the original image (value to
            # predict regardless of altitude)
            prev_points.append(center)
            # get the location of the point in the new image
            # need this in case stride is not 1. 
            new_points.append([new_y, new_x])
            # convolve
            new_im[new_y, new_x] = np.sum(np.multiply(filt, np.squeeze(im_a[i:i+filt_size, j:j+filt_size])))
            # increment to get next column in new image
            new_x += 1
        # increment to get next row in new image
        new_y +=1
    new_im = tf.keras.backend.expand_dims(new_im, axis = -1)
    print('correct if they are equal (starts at 0, so increments to final value and then exits')
    print(new_y, new_x)
    print(new_w, new_l)
    #import pdb; pdb.set_trace()
    return prev_points, new_points, new_im

#def check_dsamp(point_list, w, l, filter_size, stride)

def point_far_enough_from_sides(new_point, new_im, w, l):
    im_w = new_im.shape[0]
    im_l = new_im.shape[1]
    x = new_point[1]
    y = new_point[0]
    
    half_w = w // 2
    half_l = l //2
    # so points go from 0 to w-1 on y axis, 0 to l -1 on x axis
    
    y_cond = (y + half_w > im_w-1) or (y - half_w < 0) # if either true, then point too close to edge
    x_cond = (x + half_l > im_l-1) or (x - half_l < 0)
    #if new_im.shape[1] == 170:
    #    import pdb; pdb.set_trace()
    if y_cond or x_cond: # either x or y true, then too close to edge
        return False
    # point far enough from sides. 
    return True

# determine if a sample can be contstructed for a point in the original image
def new_alt_point(filt_s, stride, im_a, point, samp_w, samp_l, padded_convd_im):
    # first check if can perform the convolution
    # get image dimensions
    orig_w = im_a.shape[0]
    orig_l = im_a.shape[1]

    # get point
    x = int(point[1])
    y = int(point[0])
   
   # this gives you the number of points you need on either side of the point
    # to produce a full sample
    n_pts_rnl = int(stride*(samp_l //2))
    # same as above but for above and below 
    n_pts_anb = int(stride*(samp_w //2))

    # check boundary
    # note that if this works, the convolution must work, so no need to check
    # it.
    edge_loss = filt_s // 2
    #import pdb; pdb.set_trace()
    if (y - n_pts_anb < 0 + edge_loss) or (y + n_pts_anb >= orig_w-edge_loss):
        return
    elif (x - n_pts_rnl < 0 + edge_loss) or (x + n_pts_rnl >= orig_l-edge_loss):
        return
   
    # get new point
    # the way we define it, original image is altitude 1. each stride reduces
    # the image features by its value. stride 2 --> features width/2 now, same
    # with length. 
    # so we can define altitude by this new value. 
    z = stride 
    y_value = [y,x,z] # note same x,y since same point.

    # get new image sample
    # need to convolve the adjacent points too. which means there will be a lot
    # of redundant computation. 
    #import pdb; pdb.set_trace()
    x_pts = list(range(x-n_pts_rnl, n_pts_rnl+x+1, stride))
    y_pts = list(range(y-n_pts_anb, n_pts_anb+y+1, stride))
    
    x_value = padded_convd_im[y_pts,:][:,x_pts]
    assert(x_value.shape[0] == samp_w and x_value.shape[1] == samp_l)
    full_point = [x_value, y_value]
    #import pdb; pdb.set_trace()
    return full_point

def get_convolved_im(im_a, filt_s, filt):
    edge_loss = filt_s//2
    padded_convd_im = np.zeros(im_a.shape)
    for i in range(edge_loss, im_a.shape[0]-edge_loss, 1):
        for j in range(edge_loss, im_a.shape[1]-edge_loss, 1):
            point = [i,j]
            #import pdb; pdb.set_trace()
            im_part = im_a[i-edge_loss:i+edge_loss+1,j-edge_loss:j+edge_loss+1]
            new_point_val = np.sum(np.multiply(filt, np.squeeze(im_part)))
            padded_convd_im[i,j] = new_point_val

    return padded_convd_im

def points_in_image2(filt_type, filt_sizes, strides, im_a, num_req_points,
        point_list, samp_w, samp_l, return_all):
    #point_list = set(map(tuple,point_list))
    #prev_filter_dim = 1 # 1 indicates no downsampling
    count = -1 
    remaining_point_count = np.inf
   
    all_good_pts = []
    while(remaining_point_count >  num_req_points):
        prev_count = count
        count +=1
        remaining_point_count = 0
        filter_dim = filt_sizes[count]
        stride = strides[count]
        if filt_type == 'avg':
            filt = np.ones((filter_dim, filter_dim))
            filt /= (filter_dim*filter_dim)
        elif filt_type == 'gaussian':
            filt = gauss_filt(filter_dim)
    
        
        # note: point_list contains the points in the original image that
        # are in the new image. should be same length as new_center_vals. 
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        padded_convd_im = get_convolved_im(im_a, filter_dim, filt)
        point_list = list(map(tuple, point_list))
        save_pts = []
        for point in point_list:
            
            #import pdb; pdb.set_trace()
            new_pt = new_alt_point(filter_dim, stride, im_a, point, samp_w, samp_l, padded_convd_im)
            #import pdb; pdb.set_trace()
            if new_pt == None:
                continue
            remaining_point_count+=1
            save_pts.append(new_pt)
        
        all_good_pts.append(save_pts)
        print('filter size: ', filter_dim, 'stride length: ', stride, 'num reuseable points remaining: ', remaining_point_count)
        # increment filter size (to next odd number so center is still
        # centerred
    #import pdb; pdb.set_trace()
    if prev_count == -1:
        return 0, 0, 0, 0, im_a
    
    end = prev_count + 1
    if return_all:
        return filt_sizes[:end],strides[:end],all_good_pts[:end]

    return filt_sizes[prev_count], strides[prev_count], all_good_pts[prev_count]



# determine the max downsampling we can have given filter sizes, strides,
# image, etc. 
def points_in_image(filt_type, filt_sizes, strides, im_a,  num_req_points,
        point_list, w, l, return_all):
    point_list = set(map(tuple,point_list))
    #prev_filter_dim = 1 # 1 indicates no downsampling
    count = -1 
    remaining_point_count = np.inf
    new_im = []
    all_good_old_pts = []
    all_good_new_pts = []
    while(remaining_point_count >  num_req_points):
        prev_count = count
        count +=1
        remaining_point_count = 0
        filter_dim = filt_sizes[count]
        stride = strides[count]
        # note: prev_center_vals contains the points in the original image that
        # are in the new image. should be same length as new_center_vals. 
        #import pdb; pdb.set_trace()
        prev_center_vals,new_center_vals,new_im_s=strided_conv(filt_type,filter_dim, stride,im_a, point_list)
        new_im.append(new_im_s)
        #print(len(prev_center_vals), len(new_center_vals))
        prev_center_vals = list(map(tuple, prev_center_vals))
        save_old_pts = []
        save_new_pts = []
        for i, point in enumerate(prev_center_vals):
            #print(point in point_list)
            #import pdb; pdb.set_trace()

            if point in point_list and point_far_enough_from_sides(new_center_vals[i], new_im_s, w, l):
                remaining_point_count+=1
                save_old_pts.append(point)
                save_new_pts.append(new_center_vals[i])
        
        #import pdb; pdb.set_trace()
        all_good_old_pts.append(save_old_pts)
        all_good_new_pts.append(save_new_pts)
        print('filter size: ', filter_dim, 'stride length: ', stride, 'num reuseable points remaining: ', remaining_point_count)
        # increment filter size (to next odd number so center is still
        # centerred
    
    if prev_count == -1:
        return 0, 0, 0, 0, im_a
    
    end = prev_count + 1
    if return_all:
        
        return filt_sizes[:end],strides[:end],all_good_old_pts[:end],all_good_new_pts[:end],new_im[:end]

    print(prev_count)
    return filt_sizes[prev_count], strides[prev_count], all_good_old_pts[prev_count], all_good_new_pts[prev_count],new_im[prev_count]

# note w refers to vertical, l refers to horizontal
# verified it works. 
def im_samps_from_points(point_list, w, l, im_a):
    samp_list = []
    for point in point_list:
        # get upper left corner
        i = int(point[0] - (w // 2)) # along vertical # get sth like 10.0 if don't cast
        j = int(point[1] - (l //2)) # along horizontal
        new_samp = im_a[i:i+w,j:j+l,:]
        samp_list.append(new_samp)

    return samp_list

class altitude_data_wrapper:
    def __init__(self,im_a, w, l, skip_amt, val_amt, vbs, filt_type, filt_sizes,
            strides, frac_req_pts, axes, pred_type):
        # get original xy points
        if pred_type == 'xy':
            t_x,t_y,v_x,v_y = im_train_val_split_with_skips(im_a, w,
                    l,skip_amt,val_amt, vbs, get_out_xy)
        elif pred_type == 'rt':
            t_x,t_y,v_x,v_y = im_train_val_split_with_skips(im_a,
                    w,l,skip_amt,val_amt, vbs, get_out_rt)
        #import pdb; pdb.set_trace() 
        t_y = list(map(tuple,t_y))
        v_y = list(map(tuple,v_y))
       
        '''
        vo = [val for val in v_y if tuple(val) in t_y] 
        to = [val for val in t_y if tuple(val) in v_y]
        assert(len(vo) == 0)
        assert(len(to) == 0)
        '''
        all_points = []
        all_points.extend(t_y)
        #print(len(all_points))
        all_points.extend(v_y)
        # number of required points to use the altitude needed to have the same # points per altitude for balance. 
        num_req_points = int(len(all_points)*frac_req_pts)
        all_ims = []
        all_ims.extend(t_x)
        all_ims.extend(v_x)
        #print(len(all_points))
        #assert(len(all_points) == (len(t_y) + len(v_y)))
        #import pdb; pdb.set_trace()
        # get points at diff altitude based on these points
        
        #f1, fs, old_pts, new_pts, new_ims = points_in_image(filt_type,filt_sizes,strides,im_a,num_req_points,all_points,w, l,True)
        f1, fs, new_pts = points_in_image2(filt_type, filt_sizes,
                strides,im_a,num_req_points,all_points, w, l, True)

        print('filter size: ', f1[-1], 'stride length: ', fs[-1])
       
        #import pdb; pdb.set_trace()
        # adjust points to make compatible with last

        # crop point lists so same # at every altitude. 
        # get points to remove
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        
        #import pdb; pdb.set_trace()
        point_list = [val[1] for val in new_pts[-1]]
        
        point_a = np.array(point_list)
        min_y = np.min(point_a[:,0])
        max_y = np.max(point_a[:,0])
        min_x = np.min(point_a[:,1])
        max_x = np.max(point_a[:,1])

        # handle each array
        # start with original points (all_points)
        final_points = []
        orig_point_list = []
        count = 0
        for i, point in enumerate(all_points):
            y_cond = point[0] >= min_y and point[0] <= max_y
            x_cond = point[1] >= min_x and point[1] <= max_x
            if y_cond and x_cond:
                count+=1
                #import pdb; pdb.set_trace()
                orig_point_list.append([all_ims[i], [point[0], point[1], 1]]) # 1 for z
        print('count: ', count)
        final_points.append(orig_point_list) 
        #old_pts_set = set(map(tuple,new_pts[-1]))
        #print('total points: ', len(new_pts_set))
        
        #for val in old_pts:
        #    print(len(val))
        # remove points from original and subimages. remove new and old points
        # from the new arrays
        #print('prev v, t: ', len(v_y), len(t_y))
        #t_inds = [i for i in range(len(t_y)) if t_y[i] in old_pts_set]
        #t_inds = [i for i in range(len(t_y)) if t_y[i] in old_pts_set] 
        #v_inds = [i for i in range(len(v_y)) if v_y[i] in old_pts_set]
        #t_vals = [val for val in t_y if val in old_pts_set]
        #v_vals = [val for val in v_y if val in old_pts_set]
        #t_ims = [val for i, val in enumerate(t_x) if t_y[i] in old_pts_set]
        #v_ims = [val for i, val in enumerate(v_x) if v_y[i] in old_pts_set]
        
        
        #ap = [val for val in v_vals if tuple(val) in t_vals]
        #assert(len(ap) == 0)
        #assert(len(t_ims) == len(t_vals))
        #assert(len(v_ims) == len(v_vals))
        #v_inds = [i for i in range(len(v_y)) if v_y[i] in old_pts_set]
        #print('new v, t: ', len(v_inds), len(t_inds))
        #total_num_pts = len(t_vals) + len(v_vals)

        #print('num points (train + val): ', total_num_pts)
        ############## ************ ##############
        # i counts the altitudes. each has a diff set of (x,y) points -->
        # point_list
        #ap2 = [val for val in old_pts[-1] if val in old_pts[-2]]
        #new_pts = list(new_pts)
        #ap3 = [val for val in old_pts[0] if val in old_pts_set]        
        #import pdb; pdb.set_trace()

        # all based on last one so don't need it
        for i, points in enumerate(new_pts[:-1]):
            good_points = []
            for point in points:
                y_cond = point[1][0] >= min_y and point[1][0] <= max_y
                x_cond = point[1][1] >= min_x and point[1][1] <= max_x
                if y_cond and x_cond:
                    good_points.append(point) # 1 for z
            final_points.append(good_points) 
            #print('old pts filt ', f1[i], ' and stride ', fs[i], ' iter ', str(i), ' pre remove: ', len(point_list))
            #print('new pts filt', f1[i], ' and stride ', fs[i], ' iter ',str(i), ' pre remove: ', len(new_pts[i]))
            
            #import pdb; pdb.set_trace()
            #new_inds = [j for j in range(len(point_list)) if point_list[j] in old_pts_set]
            
            #new_old_pts = [val for val in point_list if val in old_pts_set]
            #new_new_pts = [val for j, val in enumerate(new_pts[i]) if point_list[j] in old_pts_set]
            #
            #print('old_pts filt', f1[i], ' and stride ', fs[i], ' iter ',str(i), ' post remove: ',len(new_old_pts))
            #print('new_pts filt', f1[i], ' and stride ', fs[i], ' iter ',str(i), ' post remove: ',len(new_new_pts))
            #old_pts[i] = new_old_pts
            #new_pts[i] = new_new_pts
            #import pdb; pdb.set_trace()
            #assert(len(old_pts[i]) == total_num_pts)
            #assert(len(new_pts[i]) == total_num_pts)

        
        assert(len(t_x) == len(t_y))
        assert(len(v_x) == len(v_y))
        # get image samps
        #im_list = []
        #for i in range(len(new_pts)):
        #    # get new im
        #    new_im_samps = im_samps_from_points(new_pts[i], w, l, new_ims[i])
        #    im_list.append(new_im_samps)
        #    #assert(len(new_im_samps) == len(old_pts[i]))
        
        # after above preprocessing, we have the following point lists:
        # cropped t_x, cropped v_x, and all the cropped new_pts
        #all_pts_new = []
        #all_pts_new.extend(t_x)
        #all_pts_new.extend(v_x)
        #assert(len(all_pts_new) == len(old_pts[0]))
        
        
        #import pdb; pdb.set_trace()     
        # divide into train and val
        # option 1 for division: 1- val -->  amt altitudes for train. each alt
        # is either train or val.
        #num_train_alts = math.floor(len(old_pts) * (1 - val_amt)) # allow at least 1 val
        num_val_alts = math.ceil(len(final_points)*val_amt) # allow at least 1 val
        
        
        # split method:
        # divide into num_val_alts sub arrays. and then take 1 entry from the
        # middle of each to make val. results in semi equally spaced val
        # arrays. 
        sub_arrays = np.array_split(final_points, num_val_alts, axis = 0) 
        
        # get middle value in each sub array, or one above true middle if
        # the length of the subarray is even
        val_pts = []
        train_pts = []
        #val_inds = []

        past_idx_count = 0
        for sub_a in sub_arrays:
            #import pdb; pdb.set_trace()
            idx = math.ceil(len(sub_a)/2)
            print('val alt added: ', idx + past_idx_count + 1)
            val_pts.extend(sub_a[idx])
            #val_ind = past_idx_count + idx - 1
            #val_inds.append(val_ind)
            #actual_inds = list(np.arange(past_idx_count, past_idx_count +len(sub_a)))
            #del actual_inds[val_ind]
            for val in sub_a[:idx]:
                train_pts.extend(val)
            for val in sub_a[idx+1:]:
                train_pts.extend(val)
            past_idx_count += len(sub_a) # use this to get the absolute position. 
        # get points. x, y are same for all alts. but z differs. 
        #import pdb; pdb.set_trace()            
        # standardize 
         

        # define values
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
        

# take an image, perform x downsamplings and make each a different altitude.
# Each sample will be centerred on a point. we then get the samples centered on
# the point at different resolutions (downsamplings). 
#def altitude_prepro():
#    #meshgrid? numpy not necessayr, we aleady have center points for previous
#    # method can we use this but then exclude points that don't scale for the
#    # altitudes? 

#l --> row length, w --> column length

#performs functions above and stores the train/test set and image. 
class data_set_wrapper:
    def __init__(self,img_path, w, l, val_frac, skip_amt, vbs, pred_type):
        im_a = img_load(img_path)
        #x_vals, y_vals = split_image_array(im_a, w,l)
        #x_t,y_t,x_v,y_v = im_train_val_split(x_vals, y_vals, val_frac)
        if pred_type == 'xy':
            x_t,y_t,x_v,y_v = im_train_val_split_with_skips(im_a, w, l,skip_amt,val_frac, vbs, get_out_xy)
        elif pred_type == 'rt':
            x_t,y_t,x_v,y_v = im_train_val_split_with_skips(im_a, w,l,skip_amt,val_frac, vbs, get_out_rt)
       
        # standardize y values (im_a done in prepro)
        if pred_type == 'xy':
            #print('not yet 2')
            y_t, y_v, w_med, w_iqr, l_med, l_iqr = standardize_xy(im_a, y_t,y_v)
            self.w_med = w_med
            self.w_iqr = w_iqr
            self.l_med = l_med
            self.l_iqr = l_iqr
        elif pred_type == 'rt':
            #print('not yet')
            y_t, y_v, r_mean, r_std, t_mean, t_std = standardize_rt(y_t,y_v)
            self.r_mean = r_mean
            self.r_std = r_std
            self.t_mean = t_mean
            self.t_std = t_std

        self.image = im_a
        self.x_train = x_t
        self.y_train = y_t
        self.x_val = x_v
        self.y_val = y_v
if __name__ == '__main__':
    apollo_img_path ='apollo_11_low_def.jpg'
    im_a = img_load(apollo_img_path)
    #print(im_a.shape)
    # for testing split_image_array
    #x_vals, y_vals = split_image_array(im_a, 31,27)
    # test x values (make sure 0, end are correct, make sure i, i+1 are shifted
    # by 1, check the edge 
    #to_image(x_vals[485])
    #print(x_vals.shape)
    
    # test y values
    # get image value at y, get x value at center, compare. 
    #print(x_vals[0], y_vals[0])
    #print(x_vals[0][y_vals[0][0],y_vals[0][1],:])
    
    # once above done, write im_train_val_split and test below. 
    #x_t,y_t,x_v,y_v = im_train_val_split(x_vals, y_vals, 0.2)
    #plot_data_set(y_t, y_v, 31, 27, im_a.shape[0], im_a.shape[1])
    
    #above but smaller shape
    #plot_data_set(y_t, y_v, 31, 27, 50, 50)
    
    # confirms working for original case. (set starts at bottom left and then
    # moves right and then up)
    #x_vals, y_vals = split_image_array(im_a, 255,501)
    #x_t,y_t,x_v,y_v = im_train_val_split(x_vals, y_vals, 0.2)
    #print(len(x_t))
    #print(len(x_v))
    #plot_data_set(y_t, y_v, 255, 501, im_a.shape[0], im_a.shape[1])
    
    # simple case to test this function
    #plot_data_set([[51,72],[35,58],[200,115]],[[81,107],[38,202],[150,150]], 31, 27,im_a.shape[0], im_a.shape[1])
    # class
    #data = data_set_wrapper(apollo_img_path, 3, 5, 0.2)
    #print(data.x_train.shape)

    # test new data generation function with skips
    w = 21
    l = 21
    skip_amt = 0
    val_amt = 0.2
    vbs = 1
    t_x, t_y, v_x, v_y = im_train_val_split_with_skips(im_a, w, l, skip_amt,val_amt, vbs, get_out_xy)
    #to_image(v_x[0], 'first_val')
    #print(v_y[0])
    #to_image(v_x[-1], 'last_val')
    #print(v_y[-1])
    #to_image(t_x[0], 'first_train')
    #print(t_y[0])
    #to_image(t_x[-1], 'last_train')
    #print(t_y[-1])
    #plot_data_set(t_y, v_y, w, l, im_a.shape[0], im_a.shape[1])

    # test strided conv
  
    
    # IDENTITY
    filt_type = 'avg'
    filt_size = 5
    stride = 4
    
    
    #to_image(new_im, 'convolved_im')
    
    #print(t_y.shape, v_y.shape)
    #import pdb; pdb.set_trace()
    samp_size = 21
    # test downsampling one
    all_points = []
    all_points.extend(list(t_y))
    all_points.extend(list(v_y))
    num_req_points = 1/10
    print('useable points in og image: ', len(all_points), 'num required useable points in new image: ', num_req_points)
    filt_sizes = [3,3,5,5,7,7,9,9, 11, 11, 13,13]
    strides = [2,3,4,5,6,7,8, 9, 10, 11, 12, 13]
    # points in image test
    #filt, strides, old_pts, new_pts, new_im = points_in_image('gaussian', filt_sizes, strides, im_a,
    #        num_req_points, all_points, w, l, False)
    #print('filter size: ',filt, 'stride length: ', strides)
    #print(len(new_pts), len(old_pts))
    #print(new_pts[0])
    #print(new_pts[-1])
    #print(new_im.shape)
    #plot_data_set(new_pts, [], 21, 21,new_im.shape[0] , new_im.shape[1])
  
    #to_image(new_im, 'new_3_im')
    a = altitude_data_wrapper(im_a, w, l, skip_amt,
            val_amt,vbs,filt_type,filt_sizes,strides,num_req_points, 'z', get_out_xy)
    #breakpoint()
    #import pdb; pdb.set_trace()
    #print(point_far_enough_from_sides([10,10], im_a, w, l))

