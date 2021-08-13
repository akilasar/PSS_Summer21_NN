#---------------------------------------------------------------#
# Description: train 2D conv lunar net models. And run basic
# evaluation
# Author: Eric Ham
# Date Last Modified: 7/15/20
#---------------------------------------------------------------#
from data_prepro import data_set_wrapper, altitude_data_wrapper
import tensorflow as tf
from ln_models import lunar_net_basic, lunar_net, lunar_resnet, naive_inception_net
from eval_ln import plot_mse, get_err, xy_converter, rt_converter, get_z_err, plot_z_vals 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import resource
from sys import stdout
import numpy as np
from data_prepro import img_load
from tensorflow import keras
from tensorflow.keras import layers
import time
import matplotlib.pyplot as plt
#tf.compat.v1.enable_eager_execution()
import keras
from keras.models import Model,load_model
checkpoint_path = 'weights'
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)
#best_path = '/Users/emham/Desktop/PSSWork/LunarNet/Results/checkpoint'
save_best = tf.keras.callbacks.ModelCheckpoint(
        filepath = os.path.join(checkpoint_path,
    '{epoch:02d}-low_lunarnet.hdf5'),
        monitor = 'val_mean_absolute_error', 
        verbose = 1,
        mode = 'min', 
        save_best_only=True)

es = tf.keras.callbacks.EarlyStopping(monitor='val_mse', min_delta=.005,mode='max', verbose=1, patience=20)

def mem(desc=''):
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stdout.write('[Mem] %s\t %dKb\n' % (desc, usage))

def set_model(model_type,ex_w,ex_l,kernel_s,stride_l,num_filt,pool_s,model_name,
        num_layers, padding_val, output_size):
    print(model_type)
    if model_type == 'lunar_net_basic':
        model = lunar_net_basic(ex_w,ex_l,num_filt,kernel_s,stride_l,pool_s,
                output_size)
    elif model_type == 'lunar_net':
        model = lunar_net(ex_w, ex_l,num_filt,num_layers,padding_val,kernel_s,stride_l,pool_s)
    elif model_type == 'lunar_resnet':
        model = lunar_resnet(ex_w,
                ex_l,num_filt,num_layers,kernel_s,stride_l,pool_s, output_size)
    elif model_type == 'naive_inception':
        #nf = [[64,64,64]] # 1, 3, 5
        nf = [[64,64,64],[128,128,128]]
        model = naive_inception_net(ex_w,ex_l,nf)

    return model
    
def checkpoint_function():
    if not os.listdir(checkpoint_path):
        return
    files_int = list()
    for i in os.listdir(checkpoint_path):
        epoch = int(i.split('-')[0])
        files_int.append(epoch)
    print(files_int)
    max_value = max(files_int)
    for i in os.listdir(checkpoint_path):
        epoch = int(i.split('-')[0])
        if epoch > max_value:
            pass
        elif epoch < max_value:
            pass
        else:
            final_file = i
    return final_file, max_value

@profile
def train_model(dg_ps, model_ps, train_ps):
    mem('Before Loading Data')
    # get data
    if dg_ps[-1] == 'xy':
        ds = data_set_wrapper(*dg_ps) # x, y only
    elif dg_ps[-1] == 'z':
        ds = altitude_data_wrapper(*dg_ps)
    
    #import pdb; pdb.set_trace()

    # z
    #ds = # x,y,z
    x_t = ds.x_train
    y_t = ds.y_train
    x_v = ds.x_val
    y_v = ds.y_val
    #pdb.set_trace()
    mem('After Loading Data')
    # get model
    model = set_model(*model_ps)
    mem('Before Compile')
    # compile model
    #model.compile(optimizer=train_ps[0], loss = train_ps[1], metrics=train_ps[2]) #metrics=[r_square])
    
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    loss_fn = tf.keras.losses.LogCosh()
    opt = tf.keras.optimizers.Adam()
    
    model.compile(optimizer=opt, loss = loss_fn, metrics=[mae_metric]) #metrics=[r_square])
    mem('After Compile, before train')
    # train model
    print('bs: ', train_ps[3])
    print(train_ps[5])
    mem('before fit')
    #history = model.fit(x=x_t, y=y_t, batch_size = train_ps[3],
    #        epochs=train_ps[4], validation_data=(x_v, y_v),
    #        shuffle=False, steps_per_epoch=None,verbose=2, callbacks=[save_best])

    mem('After train')
    # save history
    #print(checkpoint_dir)
    #print(os.listdir(checkpoint_dir))
    checkpoint_path_file = checkpoint_function()

    #if checkpoint_path_file is not None:
    path_file = checkpoint_path_file[0]
    max_value = checkpoint_path_file[1]
    print(path_file)
    print(max_value)
    model = load_model(os.path.join(checkpoint_path, path_file))
    #model.fit(GenerateInputs(X_,y_),batch_size =4, epochs=2400, verbose=1, callbacks=[checkpointer], steps_per_epoch=3, shuffle=True, initial_epoch = max_value)
    #else:
        #Model_.fit(GenerateInputs(X_,y_), epochs=2400, batch_size = 4, verbose=1, callbacks=[checkpointer], steps_per_epoch=3, shuffle=True)

    #model.load_weights(checkpoint_path)
    # save model
    #model.save('/akilatest/MODEL_'+str(model_ps[-4]) + '.h5')
    return model, ds #, history.history

#def train_iterate():
def train_model_loop(dg_ps, model_ps, train_ps):
    errors = {'train_mae': [], 'val_mae': [], 'log_cosh': [],'val_log_cosh':[], 'x_mae_val': [], 'y_mae_val': [] }
    mem('Before Loading Data')
    # get data
    ds = data_set_wrapper(*dg_ps) # x, y only
    x_t = ds.x_train
    y_t = ds.y_train
    x_v = ds.x_val
    y_v = ds.y_val
    #import pdb; pdb.set_trace()
    #mem('After Loading Data')
    # get model
    model = set_model(*model_ps)
    model.summary()
    #import pdb; pdb.set_trace()
    #model.predict(x_val)
    #mem('Before Compile')
    # compile model
    #model.compile(optimizer=train_ps[0], loss = train_ps[1], metrics=train_ps[2]) #metrics=[r_square])
    #mem('After Compile, before train')

    # # train model
    # history = model.fit(x=x_train, y=y_train, batch_size = train_ps[3],
    #             epochs=num_epochs, validation_data=(x_val, y_val),
    #             shuffle=train_ps[5], steps_per_epoch=None,verbose=2, callbacks=[save_best])

    # Instantiate an optimizer to train the model.
    optimizer = tf.keras.optimizers.Adam()
    # Instantiate a loss function.
    loss_fn = tf.keras.losses.LogCosh()

    # Prepare the metrics.
    train_mae_metric = tf.keras.metrics.MeanAbsoluteError()
    val_mae_metric = tf.keras.metrics.MeanAbsoluteError()

    # Prepare the training dataset.
    batch_size = train_ps[3]
    # jointly shuffle x_t and y_t here so that x_t[i] corresponds with y_t[i],
    # but the i's are shuffled from their original position
    #[insert code] 
    z_t = list(zip(x_t, y_t))
    np.random.shuffle(z_t)
    x_t = list(map(lambda p: p[0], z_t))
    y_t = list(map(lambda p: p[1], z_t))


    train_dataset = tf.data.Dataset.from_tensor_slices((x_t, y_t))
    #train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    train_dataset = train_dataset.batch(batch_size)
    #import pdb; pdb.set_trace()

    # Prepare the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_v, y_v))
    val_dataset = val_dataset.batch(batch_size)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            #import pdb; pdb.set_trace()
            predictions = model(x, training=True)
            
            loss_value = loss_fn(y, predictions)
            #loss_value += sum(model.losses)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_mae_metric.update_state(y, predictions)
        return predictions, loss_value

    @tf.function
    def test_step(x, y):
        val_predictions = model(x, training=False)
        val_mae_metric.update_state(y, val_predictions)
        val_loss = loss_fn(y, val_predictions)
        return val_predictions, val_loss

    # plot_data_set(y_train, y_val, w, l, im_a.shape[0], im_a.shape[1])

    mem('Before train')
    print('loop')
    all_losses = []
    epochs = train_ps[-2]
    for epoch in range(1, epochs+1):
        print("\nStart of epoch %d out of %d" % (epoch, epochs))

        #if epoch != 0:
        #    # load model weights
        #    model.load_weights(best_path)

        # Iterate over the batches of the dataset.
        loss_a = []
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            predictions, loss_value = train_step(x_batch_train, y_batch_train)
            loss_a.append(loss_value)
        print('log cosh (loss): ', np.mean(loss_a))
        errors['log_cosh'].append(np.mean(loss_a))
        #import pdb; pdb.set_trace()
        # Display metrics at the end of each epoch.
        train_mae = train_mae_metric.result()
        errors['train_mae'].append(train_mae)
        print("Training MAE over Epoch: %.4f" % (float(train_mae),))
    
        train_mae_metric.reset_states()
        
        # Run a validation loop at the end of each epoch.
        val_loss_a = []
        for x_batch_val, y_batch_val in val_dataset:
            val_predictions, val_loss  = test_step(x_batch_val, y_batch_val)
            val_loss_a.append(val_loss)
        errors['val_log_cosh'].append(np.mean(val_loss_a))
        # pdb.set_trace() 
        val_mae = val_mae_metric.result()
        errors['val_mae'].append(val_mae)
        val_mae_metric.reset_states()

        print("Validation MAE over Epoch: %.4f" % (float(val_mae),))
        # run model on data and save results
        outputs = model.predict(x_v, batch_size = 32)
        x_pred_rescale, y_pred_rescale = unstandardize_xy(outputs, ds.l_med, ds.l_iqr, ds.w_med, ds.w_iqr)
        # print('x_pred_rescale: ', x_pred_rescale)
        # print('y_pred_rescale: ', y_pred_rescale)
        # pdb.set_trace() 
        x_true_rescale, y_true_rescale = unstandardize_xy(y_v, ds.l_med, ds.l_iqr, ds.w_med, ds.w_iqr)
        # print('x_true_rescale: ', x_true_rescale)
        # print('y_true_rescale: ', y_true_rescale)
        # pdb.set_trace() 

        # if epoch % 5 == 0:
            # compute x,y error
        # x_square_err = (x_true_rescale - x_pred_rescale)**2
        # y_square_err = (y_true_rescale - y_pred_rescale)**2
        x_err = np.abs(x_true_rescale - x_pred_rescale)
        y_err = np.abs(y_true_rescale - y_pred_rescale)

        # x_mse = np.mean(x_square_err)
        # y_mse = np.mean(y_square_err)     

        # print('x mse: ', x_mse)
        # print('y mse: ', y_mse)

        x_mae = np.mean(x_err)
        y_mae = np.mean(y_err)
        errors['x_mae_val'].append(x_mae)
        errors['y_mae_val'].append(y_mae)
        print('x mae: ', x_mae)
        print('y mae: ', y_mae)
        #print('x mae std: ', np.std(x_err))
        #print('y mae std: ', np.std(y_err))
        # Reset training metrics at the end of each epoch

        # save model weights
        #model.save_weights(best_path)
        if not np.random.randint(0, 3):
            print('plotting')
            plt.figure()
            plt.plot(np.arange(len(errors['log_cosh'])), errors['log_cosh'], label='logcosh')
            plt.plot(np.arange(len(errors['val_log_cosh'])),errors['val_log_cosh'], label= 'val logcosh')
            plt.plot(np.arange(len(errors['train_mae'])), errors['train_mae'],label='train mae')
            plt.plot(np.arange(len(errors['val_mae'])), errors['val_mae'],label='val mae')
            plt.title('losses over training')
            plt.legend()
            plt.savefig('/Results/vt_loss.png')
            plt.close()
        
            plt.figure()
            plt.plot(np.arange(len(errors['val_mae'])), errors['val_mae'],label='val mae')
            plt.plot(np.arange(len(errors['x_mae_val'])),errors['x_mae_val'],label='x val mae')
            plt.plot(np.arange(len(errors['y_mae_val'])), errors['y_mae_val'],label='y val mae')
            plt.title('losses over training')
            plt.legend()
            plt.savefig('/Results/xy_mae.png')
            plt.close()



    mem('After train')
    #outputs = model.predict(x_val, batch_size = 32)
    #x_pred_rescale, y_pred_rescale = unstandardize_xy(outputs, ds.l_med, ds.l_iqr, ds.w_med, ds.w_iqr)
    #x_true_rescale, y_true_rescale = unstandardize_xy(y_val, ds.l_med, ds.l_iqr, ds.w_med, ds.w_iqr)
    #x_err = np.abs(x_true_rescale - x_pred_rescale)
    #y_err = np.abs(y_true_rescale - y_pred_rescale)

    #x_mae = np.mean(x_err)
    #y_mae = np.mean(y_err)

    #print('x mae: ', x_mae)
    #print('y mae: ', y_mae)
    #print('x mae std: ', np.std(x_err))
    #print('y mae std: ', np.std(y_err))

    return model, ds #, history

    # return model, ds, predictions, loss_value, x_mse, y_mse, errors

def unstandardize_xy(data, x_med, x_iqr, y_med, y_iqr):
    x_data = data[:, 1]
    y_data = data[:, 0]

    x_rescaled = x_data * float(x_iqr)
    x_rescaled += float(x_med)

    y_rescaled = y_data * float(y_iqr)
    y_rescaled += float(y_med)

    return x_rescaled, y_rescaled
#def train_kfold():

if __name__ == '__main__':
    
    
    
    
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default ='lunar_resnet1')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate',type=float,default=0.001)
    parser.add_argument('--model_type', type= str, default = 'lunar_resnet')
    parser.add_argument('--samp_w', type = int, default = 21)
    parser.add_argument('--samp_l', type = int, default = 21)
    #parser.add_argument('--num_hidden_layers',type=int, default = 1)
    parser.add_argument('--val_frac',type=float, default = 0.2)
    parser.add_argument('--shuffle_model',action='store_false',default=True)
    parser.add_argument('--num_filt', type = int, default = 64)
    parser.add_argument('--kernel_s',type=int,default = 3)
    parser.add_argument('--stride_l',type=int,default=1)
    parser.add_argument('--pool_s',type=int,default=3)
    parser.add_argument('--opt',type=str,default='adam')
    parser.add_argument('--loss',type=str,default = 'logcosh')
    parser.add_argument('--metrics',nargs='*',default=['mae'])
    parser.add_argument('--num_layers',type=int,default=1)
    parser.add_argument('--padding_val',type=str,default = 'valid')
    parser.add_argument('--skip_amt',type=int,default=0)
    parser.add_argument('--vbs',type=int,default=1)
    parser.add_argument('--pred_type',type=str,default='xy')
    parser.add_argument('--num_outputs',type=int, default=2)
    parser.add_argument('--filt_type',type=str,default='gaussian')
    parser.add_argument('--filt_sizes',nargs='*',default=[3,3,5,5,7,7,9,9,11,11,13,13])
    parser.add_argument('--strides',nargs='*',default=[2,3,4,5,6,7,8,9,10,11,12,13])
    parser.add_argument('--frac_req_pts',type=float, default=0.1)
    #parser.add_argument('--axes', type=str, default = 'z')
    args = parser.parse_args()
    print(args)
    
    random_value = 42
    np.random.seed(random_value)
    tf.compat.v1.random.set_random_seed(random_value)
    tf.compat.v1.reset_default_graph()

    #image_path = '/Users/emham/Desktop/PSSWork/LunarNet/apollo_11_low_def.jpg'
    image_path = os.path.join(os.getcwd(), 'apollo113km.jpg')
    save_dir = os.path.join(os.getcwd(), 'Results/') 
    #save_dir = '/Users/emham/Desktop/PSSWork/LunarNet/Results/'
    
    w = args.samp_w
    l = args.samp_l
    val_frac = args.val_frac
    model_type = args.model_type
    kernel_s = args.kernel_s
    stride_l = args.stride_l
    num_filt = args.num_filt
    pool_s = args.pool_s
    opt = args.opt
    loss = args.loss
    metrics = list(map(str,args.metrics))
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    shuffle_model = args.shuffle_model
    model_name = args.model_name
    num_layers = args.num_layers
    padding_val = args.padding_val
    pred_type = args.pred_type
    num_outputs = args.num_outputs
    filt_type = args.filt_type
    filt_sizes = args.filt_sizes
    strides = args.strides
    frac_req_pts = args.frac_req_pts
    #axes = args.axes

    #dg_ps = [image_path, w, l, val_frac]
    vbs = args.vbs
    skip_amt = args.skip_amt
    # x,y
    dg_ps = [image_path, w, l, val_frac, skip_amt, vbs, pred_type] # x,y

    # z or x,y,z depending on num_outputs, axes
    # load image
    im_a = img_load(image_path)
    # get data generator parameters
    #dg_ps = [im_a, w, l, skip_amt, val_frac, vbs, filt_type, filt_sizes,strides, 
    #    frac_req_pts, axes, pred_type] # z
    # get model parameters
    model_ps = [model_type, w, l, kernel_s, stride_l, num_filt,
            pool_s,model_name, num_layers, padding_val, num_outputs]
    # get training parameters
    train_ps = [opt, loss, metrics, batch_size, num_epochs, shuffle_model] #shuffle model should work since not data generator

    # train the model
    ln, ds = train_model(dg_ps, model_ps, train_ps)
    #ln, ds, history = train_model_loop(dg_ps, model_ps, train_ps)
    
    #print(history) 
    # evaluation
    #plot_mse(str(w), history, num_epochs, model_name,save_dir)
    # get error regardless of model outputs
    print(pred_type)
    if pred_type == 'xy': # and axes != 'z':
        get_err(ds.x_val,ds.y_val,xy_converter, ln, ds.w_med, ds.w_iqr,ds.l_med, ds.l_iqr)
    elif pred_type == 'rt': # and axes != 'z':
        get_err(ds.x_val, ds.y_val, rt_converter, ln, ds.r_mean, ds.r_std,ds.t_mean, ds.t_std)
    else:
        print('train err')
        get_z_err(ds.x_train, ds.y_train, ln, ds.z_med, ds.z_iqr)
        print('val err')
        get_z_err(ds.x_val, ds.y_val, ln, ds.z_med, ds.z_iqr)
    #ln = load_model(checkpoint_path)
    #if axes  == 'z':
     #   plot_z_vals(ds.x_train, ds.y_train, ds.x_val, ds.y_val, ds.z_iqr, ds.z_med, ln,model_name, save_dir)

    
    #cam


