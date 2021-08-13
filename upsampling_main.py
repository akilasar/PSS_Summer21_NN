#------------------------------------------------#
# Upsampling with high/low split and		 #
# recombination					 #
# Author: Eric Ham				 #
#------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt

from upsampling_utils import  diamond_square_algorithm, bilinear, cr_spline, avg_test
from scipy.signal import butter,filtfilt
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img

from PIL import Image

# recombines two images. either basic addition or weighted average
# depending on choice of beta (None or 0 <= b <=1)
def image_recomb(image1, image2, b):
    if b == None:
        return (image1 + image2)*0.5
    else:
        return image1*b + image2*(1-b)


# takes an image and upsampling method (function) as input
# applies that function to each set of 4 points in a moving window
# combines to make final upsampled image
def upsample(im, up_amt, up_func, *extra_ps):
        up_im = np.zeros((up_amt*im.shape[0] - (up_amt - 1), up_amt*im.shape[1] - (up_amt - 1)))
        sub_size = 1 + up_amt # 2 would give 3, 4 would give 5, 6 would give 7 etc.
        for i in range(im.shape[0] - 1):
            for j in range(im.shape[1] - 1):
                # upsample grid
                        # note: diamond square algorithm takes a matrix of the correct output size
                        # with the filled in corners as input 
                        # so take a square of correct output size, fill corners with adjacent 4 points
                        # in 2x2 square you are upsampling
                        if up_func == bicubic:
                            pre_im = np.zeros((up_amt*2, up_amt*2))
                            print(pre_im)
                            sub_out = up_func(pre_im, *extra_ps)    
                        else:
                                pre_im = np.zeros((sub_size, sub_size))
                                print(sub_size)
                                pre_im[0, 0] = im[i,j]
                                pre_im[0,-1] = im[i,j+1]
                                pre_im[-1, 0] = im[i+1, j]
                                pre_im[-1,-1] = im[i+1,j+1]
                                print(pre_im)
                                sub_out = up_func(pre_im, *extra_ps)

                        # add to matrix
                        up_im[i*up_amt:i*up_amt+sub_size, j*up_amt:j*up_amt+sub_size] += sub_out
                        # scale new_left + old right to get average
                        if j != 0 and i < im.shape[0] - 2: 
                            up_im[i*up_amt:i*up_amt+sub_size-1, j*up_amt] /= 2 

                        elif j != 0 and i == im.shape[0] - 2:
                            up_im[i*up_amt:i*up_amt+sub_size, j*up_amt] /= 2 

                # adjust new top row with previous bottom row 
            if i != 0:
                up_im[i*up_amt,:] /= 2
                # pixels that overlap horizontally and vertically

        return up_im

def b_filt(data, freq, order, ftype):
    b, a = butter(order, freq, btype=ftype, analog=False)
    y = filtfilt(b,a,data)
    return y

# takes image, splits into high and low frequency components, applies upsample to each
# then recombines them. returns the complete upsampled image
def full_procedure(im, up_amt, cutoff_freq, beta, order, ds_func, ds_temp):
    #import pdb; pdb.set_trace()
        low_f_im = b_filt(im, cutoff_freq, order, 'low')
        low_fu_im = upsample(low_f_im, up_amt, bilinear)

        #breakpoint()
        high_f_im = b_filt(im, cutoff_freq, order, 'high')
        high_fu_im = upsample(high_f_im, up_amt, diamond_square_algorithm, ds_func, ds_temp)

        out_im = image_recomb(low_fu_im, high_fu_im, beta) 

        return out_im

def just_bilinear(f, up_amt):
    im = Image.open(f)
    return upsample(im, up_amt, bilinear)

def just_diamond_square(im, up_amt, ds_func, ds_temp):
    return upsample(im, up_amt, diamond_square_algorithm, ds_func, ds_temp)

def just_bicubic(im, up_amt, ratio, coeff):
    return upsample(im, up_amt, bicubic, ratio, coeff)

def PIL_bilinear(f, up_amt):
    im = Image.open(f).convert('L')
    im = im.resize((im.size[0]*up_amt, im.size[1]*up_amt) , resample = Image.BILINEAR)
    return np.asarray(im)

def PIL_bicubic(f, up_amt):
    im = Image.open(f).convert('L')
    im = im.resize((im.size[0]*up_amt, im.size[1]*up_amt) , resample = Image.BICUBIC)
    return np.asarray(im)

def PIL_lanczos(f, up_amt):
    im = Image.open(f).convert('L')
    im = im.resize((im.size[0]*up_amt, im.size[1]*up_amt) , resample = Image.LANCZOS)
    return np.asarray(im)

if __name__ == '__main__':
        im = np.array(load_img('images/apollo_images/apollo113km.png', color_mode='grayscale'))
        #up_im = full_procedure(im, 4, 0.7, .7, 2, cr_spline, 25)
        #up_im = full_procedure(im, 2, 0.7, .7, 2, avg_test, 25)
        #up_im = just_bilinear(im, 4)
        #up_im = just_diamond_square(im, 2, cr_spline, 25)
        #plt.imshow(up_im)
        #plt.show()
        up_im = just_bicubic(im, 2, 1/2, -1/2)
        save_dir = 'tests/bicubic2.png'
        up_im = np.expand_dims(up_im, axis = -1)
        save_img(save_dir, up_im)


