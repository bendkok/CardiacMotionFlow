# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:56:21 2022

@author: benda
"""

import numpy as np
from tqdm import tqdm
import imageio as iio
import tensorflow as tf
from keras import backend as K
import pandas as pd
import re
import config
import os


from helpers import dice_coef
from segmentation.data_lvrv_segmentation_propagation_acdc import data_lvrv_segmentation_propagation_acdc
from segmentation.data_mesa_lvrv_segmentation_propagation_acdc import data_mesa_lvrv_segmentation_propagation_acdc
from segmentation.data_mad_ous_lvrv_segmentation_propagation_acdc import data_mad_ous_lvrv_segmentation_propagation_acdc


def dice_coef2(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true_f = tf.where(y_true > 0.5, K.ones_like(y_true), K.zeros_like(y_true))
    y_pred_f = tf.where(y_pred > 0.5, K.ones_like(y_pred), K.zeros_like(y_pred))
    # y_true = tf.where(y_true > 0.5, K.ones_like(y_true, dtype=np.int32), K.zeros_like(y_true, dtype=np.int32))
    #y_pred_f = K.flatten(y_pred)
    
    # intersection = K.sum(y_true * y_pred)
    # sum0 = K.sum(y_true) + K.sum(y_pred,)
    # # intersection = K.sum(y_true * y_pred)
    # # sum0 = K.sum(y_true, axis=[1,2]) + K.sum(y_pred)
    # return K.mean((2. * intersection + smooth) / (sum0 + smooth), axis=0)
    
    intersection = K.sum(y_true_f * y_pred_f)
    # intersection = K.sum(y_true_f + y_pred_f)/2
    # if K.sum(y_true_f)==0 and K.sum(y_pred_f)==0:
    #     print("Sum = 0")
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def key_sort_files(value):
    #from: https://stackoverflow.com/a/59175736/15147410
    """
    Extract numbers from string and return a tuple of the numeric values.
    Used to sort alpha-numerically.
    """
    return tuple(map(int, re.findall('\d+', value)))


def calculate_dice_segmentation(dataset = 'acdc', smooth=1):
    
    fold = 1 #int(sys.argv[1])
    # print('fold = {}'.format(fold))
    if fold == 0:
        mode = 'predict'
    elif fold in range(1,6):
        mode = 'val_predict'
    else:
        print('Incorrect fold')
    
    
    if dataset == 'acdc':
        
        seq_segs_pre_train = []
        for f in range(1,6):
            seq_context_imgs, seq_context_segs, seq_imgs, seq_segs = data_lvrv_segmentation_propagation_acdc(mode = 'val_predict', fold = f)
            seq_segs_pre_train += [seq_segs[se] + seq_segs[se+1] for se in range(len(seq_segs))[::2]]
        
        # seq_context_imgs, seq_context_segs, seq_imgs, seq_segs = data_lvrv_segmentation_propagation_acdc(mode = 'predict')
        # seq_segs_pre_test = [seq_segs[se] + seq_segs[se+1] for se in range(len(seq_segs))[::2]]
        
        seq_segs_pre_train = np.sort(seq_segs_pre_train)
        # seq_segs_pre_test = np.sort(seq_segs_pre_test)
        
        seq_segs_gt_train = [[a.replace('predict_lvrv2_', 'crop_2D_gt_', 1) for a in b ] for b in seq_segs_pre_train]
        seq_segs_gt_train = [[a.replace('predict_2D', 'crop_2D', 1) for a in b ] for b in seq_segs_gt_train]
        # seq_segs_gt_test = [[a.replace('predict_lvrv2_', 'crop_2D_gt_', 1) for a in b ] for b in seq_segs_pre_test]
        # seq_segs_gt_test = [[a.replace('predict_2D', 'crop_2D', 1) for a in b ] for b in seq_segs_gt_test]
        
        for_over = range(len(seq_segs_pre_train))
        subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(1, 101)]#.append("All")
        subjects.append("All mean")
        subjects.append("All")
    
        
    elif dataset == 'mesa':
        seq_context_imgs, seq_context_segs, seq_imgs, seq_segs = data_mesa_lvrv_segmentation_propagation_acdc(mode = mode, fold = fold)
        print("Not implemeted.")
        exit
    elif dataset == 'mad_ous':
        seq_context_imgs, seq_context_segs, seq_imgs, seq_segs0, gt = data_mad_ous_lvrv_segmentation_propagation_acdc(mode = 'predict', fold = fold)
        # seq_context_imgs, seq_context_segs, seq_imgs, seq_segs1, gt = data_mad_ous_lvrv_segmentation_propagation_acdc(mode = 'all', fold = fold)
        
        # seq_segs_pre_train = [item for sublist in seq_segs0 for item in sublist]
        seq_segs_pre_train = [sorted(seq_segs0[se] + seq_segs0[se+1], key=key_sort_files) for se in range(len(seq_segs0))[::2]]
        # seq_segs_pre_train = np.sort(seq_segs_pre_train)
        
        # seq_segs_gt_train = seq_segs1 #todo: test
        seq_segs_gt_train = [[a.replace('_predict_lvrv2_', '_crop_gt_', 1) for a in b ] for b in seq_segs_pre_train]
        seq_segs_gt_train = [[a.replace('_predict_lvrv_2D', '_crop_2D', 1) for a in b ] for b in seq_segs_gt_train]
        # seq_segs_gt_train = [[a.replace('predict_lvrv2_', 'crop_2D_gt_', 1) for a in b ] for b in seq_segs_pre_train] # name
        # seq_segs_gt_train = [[a.replace('MAD_OUS_predict_lvrv_2D', 'MAD_OUS_crop_2D', 1) for a in b ] for b in seq_segs_gt_train] # location
        
        for_over = np.where(gt)[0]
        out_dir = config.out_dir_mad_ous
        info_file = os.path.join(out_dir, 'MAD_OUS_info.xlsx')
        excel_data = pd.read_excel(info_file)
        subjects = pd.DataFrame(excel_data, columns=['Subject']).to_numpy().flatten()[np.where(gt==1)[0]].tolist()
        # subjects = np.array(["4","17","35","58","75","97","107","129","141","169"])[np.where(gt==1)[0]].tolist()
        subjects.append("All mean")
        subjects.append("All")
        # print(subjects)
        
        
    else:
        print("Unkown dataset.")
        raise 
        
    # seq_segs = the ground thruths
    
    predict_sequence_train = len(seq_segs_pre_train)
    # predict_sequence_test = len(seq_segs_pre_test)
    
    dice_scores_train = []
    dice_scores_test = []
    # print(f'Number of train subjects: {predict_sequence_train}. Number of test subjects: {predict_sequence_test}.')
    print(f'\nNumber of train subjects: {predict_sequence_train}.')
    
    n_zero = 0
    n_segs = 0
    
    for i in for_over:    
        segs_pre = seq_segs_pre_train[i]
        #the segs_gt is not grouped 
        segs_gt = seq_segs_gt_train[i]
        
        if len(segs_pre) != len(segs_gt):
            print(f"Not same length! {i}")
        
        curr_dice2 = []
        for frame in range(len(segs_pre)):
            
        
            segs_pre_im = np.array(iio.imread(segs_pre[frame]), dtype=float)
            segs_gt_im = np.array(iio.imread(segs_gt[frame]), dtype=float)
            
            # print(f'{i}, {frame}: segs_pre_im: {K.sum(segs_pre_im)}') if K.sum(segs_pre_im) == 0 else False
            # print(f'{i}, {frame}: segs_gt_im: {K.sum(segs_gt_im)}') if K.sum(segs_gt_im) == 0 else False
            
            if np.sum(segs_pre_im) == 0:
                # print("Sum = 0.")
                # dice = np.nan
                n_zero += 1
            # else:
            n_segs += 1
            dice = dice_coef2(segs_gt_im, segs_pre_im, smooth=smooth).numpy()

            # if not np.isnan(dice) and dice != 1.:
            if not np.isnan(dice):
                curr_dice2.append(dice)
            
        dice_scores_train.append(curr_dice2)
        
    # for i in range(predict_sequence_test):    
    #     segs_pre = seq_segs_pre_test[i]
    #     #the segs_gt is not grouped 
    #     segs_gt = seq_segs_gt_test[i]
        
    #     if len(segs_pre) != len(segs_gt):
    #         print(f"Not same length! {i}")
        
    #     segs_pre_im = np.array([iio.imread(se) for se in segs_pre], dtype=float)
    #     segs_gt_im = np.array([iio.imread(se) for se in segs_gt], dtype=float)

    #     curr_dice2 = dice_coef2(segs_gt_im, segs_pre_im).numpy()
    #     dice_scores_test.append(curr_dice2)
    
    print(f'Number of failed segs: {n_zero} of {n_segs}, {int(100*n_zero/n_segs)}%.\n')
    
    dice_flat = [item for sublist in dice_scores_train for item in sublist]
    dice_mean = [np.mean(sublist) for sublist in dice_scores_train]
    dice_scores_train.append(dice_mean)
    dice_scores_train.append(dice_flat)
    df = pd.DataFrame(dice_scores_train, index=subjects).transpose()
    print()
    des = df.describe()
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width', 1000)
    print(des)
    path = config.out_dir_mad_ous
    text_file = open(path + f"/dice_res_{dataset}.csv", "w")
    n = text_file.write(str(des))
    text_file.close()
    
        
    return dice_scores_train, subjects #, dice_scores_test


if __name__ == '__main__':
    #I think dice_coef2 is the correct one
    # dice, subjects = calculate_dice_segmentation()
    dice, subjects = calculate_dice_segmentation('mad_ous', smooth=1.0)
    
    
    
