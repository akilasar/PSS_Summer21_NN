from PIL import Image, ImageDraw
import numpy as np
import cv2
import image_slicer
import os
import math
from keras.preprocessing.image import save_img

from upsampling_main import just_bilinear, PIL_bilinear , PIL_bicubic, PIL_lanczos
from ISR.models import RDN, RRDN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w = 0
h = 0

def next_trajectory(image_dir, num_tiles, split_dir, dist_weight):
    files = []
    percentages = []
    distances = []
    coords = []
    dist_dict = {}
    center_thres = False
    center_file = None
    no_move = False

    tiles = image_slicer.slice(image_dir, num_tiles, save=False)
    image_slicer.save_tiles(tiles, directory=split_dir, prefix='img')
    
    for f in os.listdir(split_dir):
        if f.split('.')[1] in ["png", "jpg", "jpeg"]:
            files.append(f)
            im = Image.open(split_dir+f).convert('RGB')
            na = np.array(im)
            red = np.array([255,56,56]) #bbox color
            red_dot = np.dot(red.astype(np.uint32),[1,256,65536]) 
            dott = np.dot(na.astype(np.uint32),[1,256,65536]) 
            colors, count = np.unique(dott, return_counts=True)
            idx = np.where(colors == red_dot)
            w,h = im.size
            percent = count[idx]*100.0/(w*h)
            percentages.append(percent[0] if len(percent) > 0 else 0)

            split_fn = f.split('_')
            r = split_fn[1]
            c = split_fn[2].split('.')[0]
            center = math.ceil(math.sqrt(num_tiles)/2)
            new_x = (int(c)-center)*w
            new_y = (center-int(r))*h
            coords.append(np.array([new_x, new_y]))
            distances.append(math.sqrt(new_x**2 + new_y**2)) 
            if new_x == 0 and new_y == 0: #in center square
                center_file = f
                if np.array_equal(na[h//2][w//2], red) and percent > 20: #crater/obstable in center of image and more than 20% of image is covered
                    center_thres = True                
                    print('Crater in center, therefore must relocate')
                elif len(percent) > 0 and percent< 20: #crater not in center but box has less thatn 20% covered area
                   no_move = True 

    weighted = np.array(distances)*dist_weight + np.array(percentages)*(1-dist_weight)
    min_idx = np.where(weighted == min(weighted))[0][0]
    min_file = files[min_idx]
    for i in range(len(weighted)):
        dist_dict[files[i]] = weighted[i]
    
    if center_thres and min_file == center_file:
        sort_weight = np.sort(weighted)
        if sort_weight[1] - sort_weight[0] < 0.3*sort_weight[0]: #threshold for closeness of values to switch
            min_idx = np.where(weighted == sort_weight[1])[0][0]
    elif no_move:
        min_idx = np.where(weighted == dist_dict[center_file])[0][0]
    min_file = files[min_idx]
    min_coord = coords[min_idx]
    return min_coord, min_file, dist_dict

def draw_grid(f, step_count, min_c):
    image = Image.open(f).convert('RGB')
    draw = ImageDraw.Draw(image)

    w = image.width
    h = image.height
    x_start = 0
    x_end = w
    y_start = 0
    y_end = h
    step_w = int(w // step_count)
    step_h = int(h // step_count)
    
    for x in range(step_w, w-step_w+1, step_w):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill='green', width=5)

    for y in range(step_h, h-step_h+1, step_h):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill='green', width=5)

    if min_c[0] == 0 and min_c[1] == 0:
        draw.ellipse([(w//2-10, h//2-10), ((w//2+10, h//2+10))], outline= 'red', fill='red')
    else:
        draw.line([(w//2, h//2), (w//2+min_c[0], h//2-min_c[1])],fill='red', width=5)
        draw.ellipse([(w//2+min_c[0]-10, h//2-min_c[1]-10), (w//2+min_c[0]+10, h//2-min_c[1]+10)], outline= 'red', fill='red')

    del draw
    filename = "grid_"+f.split('/')[-1]
    image.save('testing/grid/' + filename)
    os.system('imgcat testing/grid/' + filename)
    return (w,h)

def mask_weights(iter_num, dist_dict, img_dir, split_dir, num_tiles):
    tiles = image_slicer.slice(img_dir, num_tiles, save=False)
    image_slicer.save_tiles(tiles, directory=split_dir, prefix='img')
    
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    c = hex2rgb('#2C99A8')
    ignore_color = (c[2], c[1], c[0])
    for tile in tiles:
        filename = tile.filename.split('/')[-1]
        image = cv2.imread(split_dir+filename)
        
        red_img  = np.full(image.shape, (0,0,255), np.uint8)
        w = dist_dict[filename]/100

        mask = (image!=ignore_color).any(-1)
        out = image.copy()
        out[mask] = image[mask] * (1-w) + red_img[mask] * w
        cv2.imwrite(split_dir+filename, out)
        
        #fused_img  = cv2.addWeighted(image, 1-w, red_img, w, 0)
        #cv2.imwrite(split_dir+filename, fused_img)
        tile.image = Image.open(split_dir+filename)
    img = image_slicer.join(tiles)
    img.save('testing/joint/' + iter_num + '_joined.png')

def cropping(w,h,move):
    t = move[1]-h//2
    b = move[1]+h//2
    l = move[0]-w//2
    r = move[0]+w//2
    if t < 0: 
        b -= t
        t = 0
    elif b > h*2:
        t -= (b-h*2)
        b = 2*h-1
    if l < 0:
        r -= l
        l = 0
    elif r > w*2:
        l -= (r-w*2)
        r = 2*w-1
    return np.array([t,b,l,r])

def simulate_trajectory(traj_fold, num_iters, f):
    if traj_fold not in os.listdir():
        os.system('mkdir ' + traj_fold)
        os.system('cp ../testing_imgs/' + f + ' ' +traj_fold)

    os.system('rm -r testing/joint/*.png')
    os.system('rm -r testing/grid/*.png')
    
    for i in range(num_iters): #loop through algo n times
        mult = num_iters - i + (1 if (num_iters-i)%2  == 0 else 2)
        num_tiles = mult**2
        os.system('rm testing/split/*.png')
        os.system('rm testing/splitimg/*.png')
        os.system('rm -r runs/detect/exp')
        os.system('rm -r runs/detect/exp2')
        os.system('rm -r testing/splitmask/*.png')
        
        #split original image
        tiles = image_slicer.slice(traj_fold +f, num_tiles, save=False)
        image_slicer.save_tiles(tiles, directory='testing/splitimg/', prefix='img')
        
        #detect on the image and preview it
        os.system('python ../low_alt_nn/obj_det/yolov5/edit_detect.py --weights ../low_alt_nn/obj_det/yolov5/runs/train/iter3/weights/best.pt --source ' + traj_fold + f + ' --img-size 350  --hide-labels --conf 0.5')
        os.system('python ../low_alt_nn/obj_det/yolov5/detect.py --weights ../low_alt_nn/obj_det/yolov5/runs/train/iter3/weights/best.pt --source ' + traj_fold + f + ' --img-size 350  --hide-labels --conf 0.5')

        #identify and display best spot in current image
        min_c, min_f, dist_dict = next_trajectory('runs/detect/exp/'+f, num_tiles, 'testing/split/', 0.038)
        img_size = draw_grid('runs/detect/exp2/' +f, mult, min_c)
        print(min_f)
        mask_weights(str(i),dist_dict, 'runs/detect/exp2/' +f, 'testing/splitmask/',num_tiles)
        os.system('imgcat testing/joint/' + str(i) + '_joined.png')
        os.system('imgcat testing/splitimg/' + min_f)

        #upsample new spot  
        new_f = PIL_bicubic(traj_fold+f, 2)
        #new_f = edsr_nn_upsample(traj_fold+f, 2)
        #new_f = isr_nn_upsample_old(traj_fold+f)
        min_c = min_c*2
        img_c = np.array([len(new_f[0])//2, len(new_f)//2])
        move = img_c + min_c*np.array([1,-1])
        w = img_size[0]
        h = img_size[1]
        locs = cropping(w,h,move) 
        new_f = new_f[locs[0]:locs[1],locs[2]:locs[3]] 
        save_dir = traj_fold + 'new_' + min_f
        #new_f = Image.fromarray(new_f).convert('L')
        new_f  = np.expand_dims(new_f, axis = -1)
        save_img(save_dir, new_f)
        f = save_dir.split('/')[1]
        os.system('imgcat '+ traj_fold +f)
        del new_f

def edsr_nn_upsample(f, num_up):
    os.system('rm up_nn/EDSR-PyTorch/test/*png*')
    os.system('cp ' + f + ' up_nn/EDSR-PyTorch/test')
    os.system('python3 up_nn/EDSR-PyTorch/src/main.py --data_test Demo --scale ' + str(num_up) + '  --test_only --save_results --dir_demo up_nn/EDSR-PyTorch/test')
    newf = 'up_nn/EDSR-PyTorch/experiment/test/results-Demo/' + f.split('/')[-1]
    return np.asarray(Image.open(newf)) 

def isr_nn_upsample_old(f):
    img = Image.open(f).convert('RGB')
    lr_img = np.array(img)
    
    #model = RDN(weights='noise-cancel')
    model = RRDN(weights='gans')
    #model = RDN(weights='psnr-small')
    #model = RDN(weights='psnr-large')
    return model.predict(lr_img) #, by_patch_of_size=10)

if __name__ == "__main__":
    #simulate_trajectory('yolotest2/', 4, 'yolotest2.png')
    #simulate_trajectory('test8/', 3, 'test8.png')
    simulate_trajectory('apollo_traj/', 2, 'apollo113km.png')
