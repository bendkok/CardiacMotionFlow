""" The main file to launch the inference of ApparentFlow-net """

import sys
sys.path.append('..')

import os
import copy
import math
import numpy as np
from PIL import Image as pil_image
# from scipy.misc import imresize
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import tensorflow as tf

from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from keras import backend as K
import re
from tqdm import tqdm

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from helpers import (
    warp_array_according_to_flow,
    flow_warped_gt_comparison_dice_loss_lvc,
    flow_warped_gt_comparison_dice_loss_lvm,
    flow_warped_gt_comparison_dice_loss_rvc,
    flow_combined_loss3,
    mean_variance_normalization5,
    elementwise_multiplication
)

from image2 import (
    array_to_img,
    ImageDataGenerator2
)
from flow.data_apparentflow import data_apparentflow
from flow.data_mesa_apparentflow import data_mesa_apparentflow
from flow.data_mad_ous_apparentflow import data_mad_ous_apparentflow

from flow.module_apparentflow_net import net_module

import config



def predict_apparentflow_net(dataset = 'acdc'):

    code_path = config.code_dir
    
    fold = 0 #int(sys.argv[1])
    print('fold = {}'.format(fold))
    if fold == 0:
        mode = 'predict'
    elif fold in range(1,6):
        mode = 'val'
    else:
        print('Incorrect fold')

    initial_lr = config.apparentflow_net_initial_lr
    decay_rate = config.apparentflow_net_decay_rate
    batch_size = config.apparentflow_net_batch_size
    input_img_size = config.apparentflow_net_input_img_size
    epochs = config.apparentflow_net_epochs

    
    ###########
    # The model
    model = net_module(input_shape=(input_img_size, input_img_size, 1), num_outputs=2)
    print('Loading model')
    model.load_weights(filepath=os.path.join(code_path, 'flow', 'model_apparentflow_net_fold{}_epoch{}.h5'.format(str(fold), str(epochs).zfill(3))) )

    
    model.compile(optimizer=Adam(lr=initial_lr), loss=flow_combined_loss3, 
        metrics=[flow_warped_gt_comparison_dice_loss_lvc, flow_warped_gt_comparison_dice_loss_lvm, flow_warped_gt_comparison_dice_loss_rvc])

    print('This model has {} parameters'.format(model.count_params()) )


    
    # Load data lists
    if dataset == 'acdc':
        img_list0, img_list1, seg_list0, seg_list1 = data_apparentflow(mode=mode, fold = fold)
    elif dataset == 'mesa':
        img_list0, img_list1, seg_list0, seg_list1 = data_mesa_apparentflow(mode=mode, fold = fold)
    elif dataset == 'mad_ous':
        img_list0, img_list1, seg_list0, seg_list1 = data_mad_ous_apparentflow(mode=mode, fold = fold)
    else:
        print("Unkown dataset.")
        raise 
        

    predict_sample = len(img_list0)
    predict_img_list = [img_list0, img_list1, seg_list0, seg_list1]

    # we create two instances with the same arguments for random transformation
    img_data_gen_args = dict(featurewise_center=False, 
                    samplewise_center=False,
                    featurewise_std_normalization=False, 
                    samplewise_std_normalization=False,
                    zca_whitening=False, 
                    zca_epsilon=1e-6,
                    rotation_range=0.,
                    width_shift_range=0., 
                    height_shift_range=0.,
                    shear_range=0., 
                    zoom_range=0.,
                    channel_shift_range=0.,
                    fill_mode='constant', 
                    cval=0.,
                    horizontal_flip=False, 
                    vertical_flip=False,
                    rescale=None, 
                    preprocessing_function=mean_variance_normalization5,
                    data_format=K.image_data_format())

    # deep copy is necessary
    mask_data_gen_args = copy.deepcopy(img_data_gen_args)
    mask_data_gen_args['preprocessing_function'] = elementwise_multiplication

    #########################
    # Generators for prediction
    print('Creating generators for prediction')
    seed = 1
    generators = []
    # The generators for the 2 inputs
    for k in range(0, 2):
        img_datagen_k = ImageDataGenerator2(**img_data_gen_args)
        img_datagen_k.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
        img_generator_k = img_datagen_k.flow_from_path_list(
            path_list=predict_img_list[k],
            target_size=(input_img_size, input_img_size), 
            pad_to_square=True,
            resize_mode='nearest', 
            histogram_based_preprocessing=False,
            clahe=False,
            color_mode='grayscale',
            class_list=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            save_period=500,
            follow_links=False)
        generators.append(img_generator_k)

    
    for k in range(2, 4):
        seg_datagen = ImageDataGenerator2(**mask_data_gen_args)
        seg_datagen.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
        seg_generator = seg_datagen.flow_from_path_list(
            path_list=predict_img_list[k],
            target_size=(input_img_size, input_img_size), 
            pad_to_square=True,
            resize_mode='nearest', 
            histogram_based_preprocessing=False,
            clahe=False,
            color_mode='grayscale',
            class_list=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            save_period=500,
            follow_links=False)
        generators.append(seg_generator)

    # Combine generators into one which yields image and masks
    predict_generator = zip(*tuple(generators))


    ###############
    # Prediction the model
    print('Start prediction')
    print('There will be {} forwards'.format( int(math.ceil(float(predict_sample)/batch_size)) ) )

    
    for j in tqdm(range( int(math.ceil(float(predict_sample)/batch_size)) )):
    # for j in range( int(math.ceil(float(predict_sample)/batch_size)) ):
        paths = predict_img_list[1][j*batch_size : min((j+1)*batch_size, predict_sample)]
        predict_batch = next(predict_generator)
        # flow: t -> ED    flow2: ED -> t
        # if predict_batch[1].shape != predict_batch[0].shape:
        #     print(predict_batch[1].shape, predict_batch[0].shape)
        flows = model.predict([predict_batch[1], predict_batch[0]], 
            batch_size=batch_size, verbose=0)
        flows2 = model.predict([predict_batch[0], predict_batch[1]], 
            batch_size=batch_size, verbose=0)
        
        
        warped_seg = warp_array_according_to_flow(predict_batch[2], flows, mode = 'nearest')
        warped_seg2 = warp_array_according_to_flow(predict_batch[3], flows2, mode = 'nearest')
        
        #what savepath to use depends on the dataset
        if dataset == 'acdc':
            orgi_dir0 = '/crop_2D/'
            new_dir0 = '/predict_2D/'
            orgi_file0 = '/crop_2D_'
            new_file0 = '/flow2_'
            
            orgi_dir1 = '/crop_2D/'
            new_dir1 = '/predict_2D/'
            orgi_file1 = '/crop_2D_'
            new_file1 = '/predict_flow_warp2_'
        elif dataset == 'mesa':
            orgi_dir0 = '/MESA_crop_2D/'
            new_dir0 = '/MESA_predict_2D/'
            orgi_file0 = '_crop_'
            new_file0 = '_flow2_'
            
            orgi_dir1 = '/MESA_crop_2D/'
            new_dir1 = '/MESA_predict_2D/'
            orgi_file1 = '_crop_'
            new_file1 = '_predict_flow_warp2_'
        elif dataset == 'mad_ous':
            orgi_dir0 = '/MAD_OUS_crop_2D/'
            new_dir0 = '/MAD_OUS_predict_2D/'
            orgi_file0 = '_crop_'
            new_file0 = '_flow2_'
            
            orgi_dir1 = '/MAD_OUS_crop_2D/'
            new_dir1 = '/MAD_OUS_predict_2D/'
            orgi_file1 = '_crop_'
            new_file1 = '_predict_flow_warp2_'
            
            # if not os.path.exists(re.sub("(.*/MESA_predict_2D/MES\d{7}).*", "\g<1>", save_path)):
            #     os.mkdir(re.sub("(.*/MESA_predict_2D/MES\d{7}).*", "\g<1>", save_path))
        
        # Save flow2   
        
        for i in range(predict_batch[0].shape[0]):
            path = paths[i]
            save_path = path.replace('\\', '/') #need this line for the following to work
            save_path = save_path.replace(orgi_dir0, new_dir0, 1)
            save_path = save_path.replace(orgi_file0, new_file0, 1)
            save_path = save_path.replace('.png', '.npy', 1)
            os.makedirs(os.path.dirname(save_path), exist_ok=True) #make sure the parent directory exists
            np.save(save_path, flows2[i], )
        

        # Resize and save the warped segmentation mask2
        # if dataset == 'acdc':
        for i in range(predict_batch[0].shape[0]):
            original_img_size = pil_image.open(paths[i]).size
            original_size = original_img_size[0]

            path = paths[i]
            warped_seg_resized2 = np.zeros((original_size, original_size, 1))
            # warped_seg_resized2[:, :, 0] = imresize(warped_seg2[i, :, :, 0], (original_size, original_size), interp = 'nearest', mode = 'F') 
            warped_seg_resized2[:, :, 0] = np.array(pil_image.fromarray(warped_seg2[i, :, :, 0], mode = 'F').resize(size=(original_size, original_size), resample = 0)) #hope this is fine, and that mode is the same

            warped_seg_resized2 = np.rint(warped_seg_resized2)
            warped_save_path2 = path.replace('\\', '/') #need this line for the following to work
            warped_save_path2 = warped_save_path2.replace(orgi_dir1, new_dir1, 1)
            warped_save_path2 = warped_save_path2.replace(orgi_file1, new_file1, 1)
            warped_seg_mask2 = array_to_img(warped_seg_resized2 * 50.0, data_format=None, scale=False)
            warped_seg_mask2.save(warped_save_path2)

        

        
        

    K.clear_session()
    print('\nApparent flow prediction is done!')


if __name__ == '__main__':
    # predict_apparentflow_net()
    # predict_apparentflow_net('mesa')
    predict_apparentflow_net('mad_ous')




