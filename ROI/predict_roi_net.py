""" The main file to launch the inference of ROI-net """


import os
import numpy as np
import scipy
import math
from PIL import Image as pil_image
import tensorflow as tf
import re
from tqdm import tqdm

from keras.models import (
    Model,
    load_model
)
from tensorflow.keras.optimizers import Adam
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from helpers import (
    dice_coef2,
    dice_coef2_loss,
    mean_variance_normalization5
)
from image2 import (
    array_to_img,
    ImageDataGenerator2
)
from ROI.data_roi_predict import data_roi_predict
from ROI.data_mesa_roi_predict import data_mesa_roi_predict
from ROI.data_mad_ous_roi_predict import data_mad_ous_roi_predict

from ROI.module_roi_net import net_module

import config
import pydicom

import sys
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


def key_sort_files(value):
    #from: https://stackoverflow.com/a/59175736/15147410
    """Extract numbers from string and return a tuple of the numeric values"""
    return tuple(map(int, re.findall('\d+', value)))


def predict_roi_net(dataset='acdc', use_info_file=True):

    code_path = config.code_dir

    initial_lr = config.roi_net_initial_lr
    batch_size = config.roi_net_batch_size
    # if dataset == 'acdc':
    input_img_size = config.roi_net_input_img_size
    # elif dataset in ['mesa']:
    #     input_img_size = config.roi_net_input_img_size_mesa
    # elif dataset == 'mad_ous':
    #     input_img_size = config.roi_net_input_img_size_mad_ous

    epochs = config.roi_net_epochs

    ###########
    # The model
    model = net_module(input_shape=(input_img_size, input_img_size, 1), num_outputs=1)

    model.load_weights(filepath=os.path.join(code_path, 'ROI', 'model_roi_net_epoch{}.h5'.format(str(epochs).zfill(3))) )

    model.compile(optimizer=Adam(lr=initial_lr), loss=dice_coef2_loss, 
        metrics=[dice_coef2])

    ######
    # Data
    if dataset == 'acdc':
        predict_img_list, predict_gt_list = data_roi_predict()
    elif dataset in ['mesa', 'MESA']:
        dataset_name = 'MESA'
        predict_img_list, predict_gt_list, subject_dir_list, original_2D_paths, gt = data_mesa_roi_predict(use_info_file, True)
        subject_dir_list = sorted(subject_dir_list, key=key_sort_files)
    elif dataset in ['mad_ous', 'MAD_OUS']:
        dataset_name = 'MAD_OUS'
        predict_img_list, predict_gt_list, subject_dir_list, original_2D_paths, gt = data_mad_ous_roi_predict(use_info_file, True)
        subject_dir_list = sorted(subject_dir_list, key=key_sort_files)
    else:
        print("Unknown dataset.")
        raise
        
    
    predict_img_list = sorted(predict_img_list, key=key_sort_files)
    predict_gt_list = sorted(predict_gt_list, key=key_sort_files)

    predict_sample = len(predict_img_list)

    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=False, 
                    samplewise_center=False,
                    featurewise_std_normalization=False, 
                    samplewise_std_normalization=False,
                    zca_whitening=False, 
                    zca_epsilon=1e-6,
                    rotation_range=0.0,
                    width_shift_range=0.0, 
                    height_shift_range=0.0,
                    shear_range=0., 
                    zoom_range=0.0,
                    channel_shift_range=0.,
                    fill_mode='constant', 
                    cval=0.,
                    horizontal_flip=False, 
                    vertical_flip=False,
                    rescale=None, 
                    preprocessing_function=mean_variance_normalization5,
                    data_format=K.image_data_format())


    ###########################
    # Generators for predicting
    image_datagen = ImageDataGenerator2(**data_gen_args)

    seed = 1
    image_datagen.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)

    image_generator = image_datagen.flow_from_path_list(
            path_list=predict_img_list,
            target_size=(input_img_size, input_img_size), 
            pad_to_square=True,
            resize_mode='nearest', 
            histogram_based_preprocessing=True,
            clahe=True,
            color_mode='grayscale',
            class_list=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            save_period=100,
            follow_links=False)


    print('Start prediction')
    print('There will be {} batches with batch-size {}'.format(int(math.ceil(float(predict_sample) / batch_size)), batch_size) )
    
    sizes_ = []
    for i in tqdm(range(int(math.ceil(float(predict_sample) / batch_size)) ), file=sys.stdout):
        with nostdout():
            print('batch {}'.format(i) )
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, predict_sample)
            img_list_batch = predict_img_list[start_idx:end_idx]
    
            batch_img = next(image_generator) 
    
            predict_masks = model.predict(batch_img, 
                                          batch_size=batch_size, 
                                          verbose=0)
            binarized_predict_masks = np.where(predict_masks >= 0.5, 1.0, 0.0)
            
            sizes = []
            for j in range(len(img_list_batch)):
                img_path = img_list_batch[j]
                # print(img_path)
                img_path = img_path.replace('\\', '/')
                # print(img_path)
                # exit
                if dataset == 'acdc':
                    img_size = pil_image.open(img_path).size
                    h = img_size[0]
                    w = img_size[1]
                elif dataset in ['mesa', 'mad_ous']:
                    # subject_data = pydicom.read_file(img_path, force=True) 
                    # h = subject_data.Rows #img_size[0]
                    # w = subject_data.Columns #img_size[1]
                    img_size = pil_image.open(img_path).size
                    h = img_size[0]
                    w = img_size[1]
                size = max(h, w)
                if j == 0:
                    sizes.append([h,w])
                
                # reshape and crop the predicted mask to the original size
                # print(binarized_predict_masks.shape)
                # print(binarized_predict_masks[j,:size,:size].shape)
                # mask = np.reshape(binarized_predict_masks[j,:size,:size], newshape=(size, size))
                mask = np.reshape(binarized_predict_masks[j], newshape=(input_img_size, input_img_size))
                # print(mask.shape)
                # print(size)
                # resized_mask = scipy.misc.imresize(mask, size=(size, size), interp='nearest')/255.0
                resized_mask = np.array(pil_image.fromarray(mask).resize(size=(size, size), resample=0)) / 255.0 # I think this does the same, not sure
                cropped_resized_mask = resized_mask[((size-w)//2):((size-w)//2 + w), 
                                                    ((size-h)//2):((size-h)//2 + h)]
                cropped_resized_mask = np.reshape(cropped_resized_mask, newshape=(w, h, 1))
                
                
    
                if dataset == 'acdc':
                    predicted_mask_path = img_path.replace('original_2D', 'mask_original_2D', 2)
                    
                    # save txt file
                    predicted_mask_txt_path = predicted_mask_path.replace('.png', '.txt', 1)
                    np.savetxt(predicted_mask_txt_path, cropped_resized_mask.reshape((cropped_resized_mask.shape[0],-1)), fmt='%.6f') #hope this is fine
    
                    # save image
                    cropped_resized_mask_img = array_to_img(cropped_resized_mask,
                                                            data_format=None, 
                                                            scale=True)
                    cropped_resized_mask_img.save(predicted_mask_path)
    
                # elif dataset == 'mesa':
                #     predicted_mask_path = re.sub("MESA_set1_sorted/(MES0\d{6}).*/([0-9]{1,3}_)(sliceloc.*)", 'MESA_mask_original_2D/\g<1>/\g<2>mask_\g<3>', img_path)
                #     # print(predicted_mask_path)
                #     # exit
                    
                #     # save txt file
                #     predicted_mask_txt_path = predicted_mask_path + '.txt'
                #     np.savetxt(predicted_mask_txt_path, cropped_resized_mask.reshape((cropped_resized_mask.shape[0],-1)), fmt='%.6f') #hope this is fine
    
                #     # save image
                #     cropped_resized_mask_img = array_to_img(cropped_resized_mask,
                #                                             data_format=None, 
                #                                             scale=True)
                #     cropped_resized_mask_img.save(predicted_mask_path + '.png')
                    
                elif dataset in ['mesa','mad_ous']:
                    # predicted_mask_path = re.sub("MAD_OUS_sorted/([0-9]+)/cine.+/([0-9]{1,3}_)_sliceloc_(.*)", 'MAD_OUS_mask_original_2D/\g<1>/\g<2>mask_\g<3>', img_path)
                    predicted_mask_path = img_path.replace(f'{dataset_name}_preprocess_original_2D', f'{dataset_name}_mask_original_2D')
                    predicted_mask_path = predicted_mask_path.replace('sliceloc', 'mask')
                    # print(img_path)
                    # print(predicted_mask_path)
                    # exit
                    
                    # save txt file
                    predicted_mask_txt_path = predicted_mask_path.replace('.png', '.txt')
                    np.savetxt(predicted_mask_txt_path, cropped_resized_mask.reshape((cropped_resized_mask.shape[0],-1)), fmt='%.6f') #hope this is fine
    
                    # save image
                    cropped_resized_mask_img = array_to_img(cropped_resized_mask,
                                                            data_format=None, 
                                                            scale=True)
                    cropped_resized_mask_img.save(predicted_mask_path)

            sizes_.append(sizes)
            
    print(sizes_)

if __name__ == '__main__':
    predict_roi_net()
    # predict_roi_net('mesa')
    # predict_roi_net('mad_ous')


