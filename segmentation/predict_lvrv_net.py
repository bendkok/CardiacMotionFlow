""" The main file to launch the inference of LVRV-net """

import sys
sys.path.append('..')

import os
import copy
import numpy as np
import numpy.ma as ma
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

# from scipy.misc import imresize
from PIL import Image as pil_image
import tensorflow as tf
import nibabel as nib
import pandas as pd
import re
import matplotlib.pyplot as plt 
import cv2 as cv

from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from keras import backend as K
from tqdm import tqdm
# import tqdm

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from helpers import (
    dice_coef5_0,
    dice_coef5_1,
    dice_coef5_2,
    dice_coef5_3,
    dice_coef5,
    dice_coef5_loss,
    mean_variance_normalization5,
    elementwise_multiplication,
    keep_largest_components,
    touch_length_count,
    number_of_components,
    second_largest_component_count
)

from image2 import (
    array_to_img,
    ImageDataGenerator2
)

from segmentation.data_lvrv_segmentation_propagation_acdc import data_lvrv_segmentation_propagation_acdc
from segmentation.data_lvrv_segmentation_propagation_mesa import data_lvrv_segmentation_propagation_mesa
from segmentation.data_lvrv_segmentation_propagation_mad_ous import data_lvrv_segmentation_propagation_mad_ous

from segmentation.module_lvrv_net import net_module

import config

import contextlib

#source: https://stackoverflow.com/a/37243211/15147410
class DummyFile(object):
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)
            
    def flush(self):
        pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout


def predict_lvrv_net(dataset = 'acdc', fold = 0):

    code_path = config.code_dir
    
    # fold = 0 #int(sys.argv[1])
    print('fold = {}'.format(fold))
    if fold == 0:
        mode = 'predict'
    elif fold in range(1,6):
        mode = 'val_predict'
    else:
        print('Incorrect fold')

    initial_lr = config.lvrv_net_initial_lr
    input_img_size = config.lvrv_net_input_img_size
    epochs = config.lvrv_net_epochs
    batch_size = 1

    ###########
    # The model
    model = net_module(input_shape=(input_img_size, input_img_size, 1), num_outputs=4)
    print('Loading model')

    
    model.load_weights(filepath=os.path.join(code_path, 'segmentation', 'model_lvrv_net_finetune_fold{}_epoch{}.h5'.format(str(fold), str(epochs).zfill(3))) )
    

    model.compile(optimizer=Adam(lr=initial_lr),loss=dice_coef5_loss, 
        metrics=[dice_coef5, dice_coef5_0, dice_coef5_1, dice_coef5_2, dice_coef5_3])

    print('This model has {} parameters'.format(model.count_params()) )

    if dataset == 'acdc':
        seq_context_imgs, seq_context_segs, seq_imgs, seq_segs = data_lvrv_segmentation_propagation_acdc(mode = mode, fold = fold)
        make_crop_nii = False
    elif dataset == 'mesa':
        seq_context_imgs, seq_context_segs, seq_imgs, seq_segs, gt = data_lvrv_segmentation_propagation_mesa(mode = mode, fold = fold)
        dataset_name = 'MESA'
        out_dir = config.out_dir_mesa
        info_file = os.path.join(out_dir, 'MESA_info.xlsx')
        excel_data = pd.read_excel(info_file)
        data = pd.DataFrame(excel_data, columns=['Subject', 'Direcory', 'Filepath', 'ED', 'ES', 'Slices', 'Instants'])
        subjects = data.Subject.to_numpy(dtype=str)
        subjects = [[subjects[i],subjects[i]] for i in range(len(subjects))]
        subjects = [item for sublist in subjects for item in sublist]
        frame_names0 = data.ED.to_numpy(dtype=int)
        frame_names1 = data.ES.to_numpy(dtype=int)
        frame_names = [[frame_names0[i],frame_names1[i]] for i in range(len(frame_names0))]
        frame_names = [item for sublist in frame_names for item in sublist]
        make_crop_nii = True
    elif dataset == 'mad_ous':
        seq_context_imgs, seq_context_segs, seq_imgs, seq_segs, gt = data_lvrv_segmentation_propagation_mad_ous(mode = mode, fold = fold)
        dataset_name = 'MAD_OUS'
        out_dir = config.out_dir_mad_ous
        info_file = os.path.join(out_dir, 'MAD_OUS_info.xlsx')
        excel_data = pd.read_excel(info_file)
        data = pd.DataFrame(excel_data, columns=['Subject', 'Direcory', 'Filepath', 'ED', 'ES', 'Slices', 'Instants'])
        subjects = data.Subject.to_numpy(dtype=str)
        subjects = [[subjects[i],subjects[i]] for i in range(len(subjects))]
        subjects = [item for sublist in subjects for item in sublist]
        frame_names0 = data.ED.to_numpy(dtype=int)
        frame_names1 = data.ES.to_numpy(dtype=int)
        frame_names = [[frame_names0[i],frame_names1[i]] for i in range(len(frame_names0))]
        frame_names = [item for sublist in frame_names for item in sublist]
        make_crop_nii = True
    else:
        print("Unkown dataset.")
        raise 
    


    predict_sequence = len(seq_imgs)

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
                    horizontal_flip= False, 
                    vertical_flip=False,
                    rescale=None, 
                    preprocessing_function=mean_variance_normalization5,
                    data_format=K.image_data_format())

    # deep copy is necessary
    mask_data_gen_args = copy.deepcopy(img_data_gen_args)
    mask_data_gen_args['preprocessing_function'] = elementwise_multiplication

    #########################
    # Generators for training
    print('Creating generators for prediction')
    image_context_datagen = ImageDataGenerator2(**img_data_gen_args)
    image_datagen = ImageDataGenerator2(**img_data_gen_args)
    mask_context_datagen = ImageDataGenerator2(**mask_data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_context_datagen.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
    image_datagen.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
    mask_context_datagen.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)


    print('Start prediction')
    print('There will be {} sequences'.format(predict_sequence) )
    failed_segs = 0
    att = 0
    
    for i in tqdm(range(predict_sequence), file=sys.stdout):
        with nostdout():
            print('\nSequence # {}'.format(i) )
    
            # The lists fot the sequence
            context_imgs = seq_context_imgs[i]
            context_segs = seq_context_segs[i]
            imgs = seq_imgs[i]
            segs = seq_segs[i]
    
    
            image_context_generator = image_context_datagen.flow_from_path_list(
                path_list=context_imgs,
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
                do_print_index_array=True,
                save_to_dir=None,
                save_prefix='',
                save_format='png',
                save_period=500,
                follow_links=False,
                do_print_found=True)
    
            image_generator = image_datagen.flow_from_path_list(
                path_list=imgs,
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
    
            mask_context_generator = mask_context_datagen.flow_from_path_list(
                path_list=context_segs,
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
    
    
            # Combine generators into one which yields image and masks
            predict_generator = zip(image_context_generator, image_generator, mask_context_generator)
    
            
            img_size = pil_image.open(imgs[0]).size
            size = img_size[0]
    
            image_data = np.zeros((size, size, len(imgs)))
    
            for j in range(len(imgs)):
                
                img_context, img, mask_context = next(predict_generator)
                
                masks = model.predict([img_context, img, mask_context], 
                    batch_size=batch_size, verbose=0)
    
                masks = np.reshape(masks, newshape=(input_img_size, input_img_size, 4))
                masks_resized = np.zeros((size, size, 4))
                for c in range(4):
                    # masks_resized[:, :, c] = imresize(masks[:, :, c], (size, size), interp='bilinear')
                    masks_resized[:, :, c] = np.array(pil_image.fromarray(masks[:, :, c]).resize(size=(size, size), resample=2)) #changed this because the old verison was depricated. might not do exactly the same
                prediction_resized = np.argmax(masks_resized, axis=-1)
                prediction_resized = np.reshape(prediction_resized, newshape=(size, size, 1))
    
                # Check whether the prediction is successful
                have_lvc = (1 in prediction_resized)
                have_lvm = (2 in prediction_resized)
                lvc_touch_background_length = touch_length_count(prediction_resized, size, size, 1, 0)
                lvc_touch_lvm_length = touch_length_count(prediction_resized, size, size, 1, 2)
                lvc_touch_rvc_length = touch_length_count(prediction_resized, size, size, 1, 3)
    
                # lvc_second_largest_component_count = second_largest_component_count(prediction_resized, 1)
                # lvm_second_largest_component_count = second_largest_component_count(prediction_resized, 2)
                # rvc_second_largest_component_count = second_largest_component_count(prediction_resized, 3)
    
                
                # success = have_lvc and \
                success = have_lvm and \
                    ((lvc_touch_background_length + lvc_touch_rvc_length) <= 0.5 * lvc_touch_lvm_length)
    
                att += 1
                if not success:
                    prediction_resized = 0 * prediction_resized
                    print('Unsuccessful segmentation for {}'.format(imgs[j]))
                    failed_segs += 1
                    # if np.max(prediction_resized) > 0:
                    #     prediction_resized #todo: this never triggers, thus the error must be on line 252 or 255-261
                else:
                    prediction_resized = keep_largest_components(prediction_resized, keep_values=[1, 2, 3], values=[1, 2, 3])
                    # if np.max(prediction_resized) > 2:
                    #     prediction_resized
                
    
                # save txt file
                prediction_path = segs[j]   
                prediction_txt_path = prediction_path.replace('.png', '.txt', 1)
                # np.savetxt(prediction_txt_path, prediction_resized, fmt='%.6f')
                os.makedirs(os.path.dirname(prediction_txt_path), exist_ok=True) #make sure the parent directory exists
                np.savetxt(prediction_txt_path, prediction_resized.reshape((prediction_resized.shape[0],-1)), fmt='%.6f') #hope this is fine
    
                # save image
                prediction_img = array_to_img(prediction_resized * 50.0,
                                              data_format=None, 
                                              scale=False)
                prediction_img.save(prediction_path)
                
                #creates an overlayed image
                # plt.imshow(img[0,:,:,0],'gray', interpolation='none')
                # plt.imshow(ma.masked_array(np.array(prediction_img)>0, prediction_img), 'jet', interpolation='none', alpha=0.5)
                # plt.show()                
                
                # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                # # overlayed = pil_image.blend(pil_image.fromarray(img[0,:,:,0]).resize(size=(size, size)).convert('L'), prediction_img.convert('L'), .7)
                # overlayed = pil_image.blend(pil_image.fromarray(clahe.apply(img[0,:,:,0].astype('uint8'))).resize(size=(size, size)).convert('LA'), prediction_img.convert('LA'), .7)
                # overlay_path = prediction_path.replace('_predict_', '_comp_')
                # os.makedirs(re.sub('(.*)\\\\.*', '\g<1>/', overlay_path), exist_ok=True)
                # overlayed.save(overlay_path, 'PNG')
                
                #saves in large array that can be saved as .nii.gz
                image_data[:,:,j] = prediction_resized[:,:,0] 
            
            if make_crop_nii:                            
                label_file = os.path.join(out_dir, f'{dataset_name}_predict_lvrv_2D',
                                'predict_lvrv2_{}_frame{}.nii.gz'.format(subjects[i],str(frame_names[i]+1).zfill(2)))
                nib.save(nib.Nifti1Image(np.rot90(image_data[:, ::-1, :],-1), np.eye(4)), label_file)
                # nib.save(nib.Nifti1Image(image_data, np.eye(4)), label_file)


    K.clear_session()
    print("Segmentation prediction done!")
    # print(f"There were {failed_segs} failed segmentations out of a total of {predict_sequence*len(imgs)}.")
    # print(f"{failed_segs} out of {att} ({100*failed_segs/att}%) segmentations were unsuccessful.")
    print(f"{failed_segs} out of {att} ({int(100*failed_segs/att)}%) segmentations were unsuccessful.")



if __name__ == '__main__':
    for i in range(1,6):
        predict_lvrv_net(fold = i)
    # predict_lvrv_net("mesa")
    # predict_lvrv_net("mad_ous")




