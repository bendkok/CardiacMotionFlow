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

from helpers import dice_coef
from segmentation.data_lvrv_segmentation_propagation_acdc import data_lvrv_segmentation_propagation_acdc
from segmentation.data_mesa_lvrv_segmentation_propagation_acdc import data_mesa_lvrv_segmentation_propagation_acdc
from segmentation.data_mad_ous_lvrv_segmentation_propagation_acdc import data_mad_ous_lvrv_segmentation_propagation_acdc


def dice_coef2(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true = tf.where(y_true > 0.5, K.ones_like(y_true), K.zeros_like(y_true))
    y_pred = tf.where(y_pred > 0.5, K.ones_like(y_pred), K.zeros_like(y_pred))
    # y_true = tf.where(y_true > 0.5, K.ones_like(y_true, dtype=np.int32), K.zeros_like(y_true, dtype=np.int32))
    #y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true * y_pred, axis=[1,2])
    sum0 = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    # intersection = K.sum(y_true * y_pred)
    # sum0 = K.sum(y_true, axis=[1,2]) + K.sum(y_pred)
    return K.mean((2. * intersection + smooth) / (sum0 + smooth), axis=0)


def calculate_dice_segmentation(dataset = 'acdc'):
    
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
        #hopefully this works
    elif dataset == 'mesa':
        seq_context_imgs, seq_context_segs, seq_imgs, seq_segs = data_mesa_lvrv_segmentation_propagation_acdc(mode = mode, fold = fold)
    elif dataset == 'mad_ous':
        seq_context_imgs, seq_context_segs, seq_imgs, seq_segs = data_mad_ous_lvrv_segmentation_propagation_acdc(mode = mode, fold = fold)
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
    for i in range(predict_sequence_train):    
        segs_pre = seq_segs_pre_train[i]
        #the segs_gt is not grouped 
        segs_gt = seq_segs_gt_train[i]
        
        if len(segs_pre) != len(segs_gt):
            print(f"Not same length! {i}")
        
        segs_pre_im = np.array([iio.imread(se) for se in segs_pre], dtype=float)
        segs_gt_im = np.array([iio.imread(se) for se in segs_gt], dtype=float)

        curr_dice2 = dice_coef2(segs_gt_im, segs_pre_im).numpy()
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
        
    return [dice_scores_train]#, dice_scores_test


if __name__ == '__main__':
    #I think dice_coef2 is the correct one
    dice = calculate_dice_segmentation()
    print(dice[0])    
    df = pd.DataFrame(dice[0])
    print()
    print(df.describe())
    
    # print()
    # print(dice[1])    
    # df = pd.DataFrame(dice[1])
    # print(df.describe())
    # print(np.where(np.array(dice[0]) - np.array(dice[1]) < 1e-1)[0], len(np.where(np.array(dice[0]) - np.array(dice[1]) < 1e-1)[0]))
    # print(np.where(np.array(dice[0]) - np.array(dice[1]) < 1e-2)[0], len(np.where(np.array(dice[0]) - np.array(dice[1]) < 1e-2)[0]))
    # print(np.where(np.array(dice[0]) - np.array(dice[1]) < 1e-3)[0], len(np.where(np.array(dice[0]) - np.array(dice[1]) < 1e-3)[0]))
    # print()
    # print(np.array(dice[1]) / np.array(dice[0]))
    # print(len(dice[0]), len(dice[1]))
    
