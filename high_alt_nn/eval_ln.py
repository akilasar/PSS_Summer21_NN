#-----------------------------------------------------------------
# Code for evaluating the lunar net
# Author: Eric Ham
# Date Last Modified: 7/15/20
#-----------------------------------------------------------------
import matplotlib.pyplot as plt
from keras import models
import numpy as np
import tensorflow as tf

random_value = 42
np.random.seed(random_value)
tf.compat.v1.random.set_random_seed(random_value)
tf.compat.v1.reset_default_graph()

def plot_mse(name, history, identifier, save_dir_i):
    plt.figure(figsize=(22,10))
    
    val_mse = history['val_loss']
    train_mse = history['loss']
    num_epochs = len(train_mse) 
    plt.plot(range(num_epochs), val_mse, 'bo', linestyle='-', label= 'Val Loss', color='orange')
    plt.plot(range(num_epochs), train_mse, 'bo', linestyle='-', label = 'Train Loss', color='blue')
    plt.title('Val and Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(save_dir_i + name +'_mse_plot.png')
    plt.close()

# for model that directly predicts x and y. just output these values.
def xy_converter(model_out, y_med, y_iqr, x_med, x_iqr):
    # get outputs
    y, x = model_out[:,0], model_out[:,1]
    
    # undo standardization
    y *= y_iqr
    y += y_med
    x *= x_iqr
    x += x_med
    out = np.zeros((len(x),2))
    out[:,0] = y
    out[:,1] = x
    return out

# have to convert from polar to x,y
def rt_converter(model_out, r_mean, r_std, t_mean, t_std):
    r,t = model_out[:,0], model_out[:,1]
    r *= r_std
    r += r_mean

    t *= t_std
    t += t_mean

    x = r * np.cos(t)
    y = r * np.sin(t)
    out = np.zeros((len(x),2))
    out[:,0] = y
    out[:,1] = x
    return out

# convert model outputs to error using model specific converter
def get_err(v_in, v_out,converter, model, *converter_args):
    # run model on data and save results
    outputs = model.predict(v_in, batch_size = 32)
    # convert true outputs and model outputs to x,y position
    #import pdb; pdb.set_trace()
    xy_out = converter(outputs, *converter_args)
    x = xy_out[:,1]
    y = xy_out[:,0]
    xy_out_t = converter(v_out,*converter_args)
    x_t = xy_out_t[:,1]
    y_t = xy_out_t[:,0]
    # compute x,y error
    x_square_err = (x-x_t)**2
    y_square_err = (y-y_t)**2
    x_err = np.abs(x-x_t)
    y_err = np.abs(y-y_t)

    x_mse = np.mean(x_square_err)
    y_mse = np.mean(y_square_err)

    x_mae = np.mean(x_err)
    y_mae = np.mean(y_err)
    
    print('x mse: ', x_mse)
    print('y mse: ', y_mse)

    print('x mae: ',x_mae)
    print('y_mae: ', y_mae)
    print('x mae std: ', np.std(x_err))
    print('y mae std: ', np.std(y_err))

def get_z_err(v_in, v_out,model, z_med, z_iqr):
    z = model.predict(v_in, batch_size = 32)
    z *= z_iqr
    z += z_med

    z_t = v_out
    z_t *= z_iqr
    z_t += z_med

    z_square_err = (z-z_t)**2
    z_err = np.abs(z-z_t)

    z_mse = np.mean(z_square_err)

    z_mae = np.mean(z_err)
    
    print('z mse: ', z_mse)

    print('z mae: ',z_mae)
    print('z mae std: ', np.std(z_err))

def plot_z_vals(t_in, t_out, v_in, v_out, z_iqr, z_med, model,identifier, save_dir_i):
    train_zt = t_out
    train_zt *= z_iqr
    train_zt += z_med
    
            
    train_zp = model.predict(t_in, batch_size = 32)
    train_zp *= z_iqr
    train_zp += z_med

    val_zt = v_out
    val_zt *= z_iqr
    val_zt += z_med

    val_zp = model.predict(v_in, batch_size = 32)
    val_zp *= z_iqr
    val_zp += z_med
    #import pdb; pdb.set_trace()
    # get pred vals that line up
    true_alts = set(list(train_zt))
    train_boxes = []
    val_boxes = []
     
    val_alts = set(list(val_zt))
    #import pdb; pdb.set_trace()
    box_dict = {}
    box_color = {}
    for alt in true_alts:
        train_a = list(map(float,[train_zp[i] for i in range(len(train_zt)) if train_zt[i] == alt]))
        #train_boxes.append(list(map(int, train_a)))
        box_dict[alt] = train_a
        box_color[alt] = '#0000FF'

    for alt in val_alts:
        val_a = list(map(float,[val_zp[i] for i in range(len(val_zt)) if val_zt[i] == alt]))
        #val_boxes.append(list(map(int, val_a)))
        box_dict[alt] = val_a
        box_color[alt] = '#00FF00'
    #import pdb; pdb.set_trace()
    sort_box = dict(sorted(box_dict.items()))
    sort_color = dict(sorted(box_color.items()))
    
    alts = [k for k, v in sort_box.items()]
    box_list = [v for k, v in sort_box.items()]
    box_color = [v for k, v in sort_color.items()]

    #val_boxes.append([val for val in val_zp if val == alt])
    #ticks = list(true_alts)
    #val_ticks = list(val_alts)
    #plt.figure()
    fig, ax = plt.subplots()
    bp = ax.boxplot(box_list, positions = alts, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_color):
        patch.set_facecolor(color)
    plt.title('Val and Train Box Plot')

    #plt.plot(train_zt, train_zp,'bo', label='train set')
    #plt.plot(val_zt, val_zp,'ro', label='val set')
    #ax.boxplot(train_boxes, positions=ticks, labels=ticks)
    #ax.boxplot(val_boxes, positions = val_ticks, labels=val_ticks)
    #plt.boxplot(train_boxes, positions=ticks)
    #plt.xlabel('True Altitude')
    #plt.ylabel('Predicted Altitude')
    #plt.title('Pred vs True Altitude For Val and Train Sets')
    #plt.legend()
    plt.savefig(save_dir_i + identifier + '/tp_alt_plot.png')
    plt.close()

    
#def cam():
