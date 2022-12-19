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
import matplotlib.pyplot as plt


from helpers import dice_coef
from segmentation.data_lvrv_segmentation_propagation_acdc import data_lvrv_segmentation_propagation_acdc
from segmentation.data_lvrv_segmentation_propagation_mesa import data_lvrv_segmentation_propagation_mesa
from segmentation.data_lvrv_segmentation_propagation_mad_ous import data_lvrv_segmentation_propagation_mad_ous


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
    # if np.sum(intersection) > 0:
    #     print()
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


def calculate_dice_segmentation(dataset = 'acdc', smooth=1, do_latex_table=True):
    
    fold = 0 #int(sys.argv[1])
    # print('fold = {}'.format(fold))
    if fold == 0:
        mode = 'predict'
    elif fold in range(1,6):
        mode = 'val_predict'
    else:
        print('Incorrect fold')
    
    
    if dataset == 'acdc':
        
        dataset_name = 'ACDC'
        out_dir = config.code_dir
        
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
        # seq_context_imgs, seq_context_segs, seq_imgs, seq_segs = data_lvrv_segmentation_propagation_mesa(mode = mode, fold = fold)
        # print("Not implemeted.")
        # exit
        
        dataset_name = 'MESA'
        
        out_dir = config.out_dir_mesa
        info_file = os.path.join(out_dir, 'MESA_info.xlsx')
        excel_data = pd.read_excel(info_file)
        
        first = pd.DataFrame(excel_data, columns=['First']).to_numpy().flatten() # [0]*10  
        last  = pd.DataFrame(excel_data, columns=['Last']).to_numpy().flatten() # [-1]*10
        ind = [[first[i]] + [last[i]] for i in range(len(first))]
        ind = [item for sublist in ind for item in sublist]
        # ind = [first[i] + last[i] for i in range(len(first))]
        
        seq_context_imgs, seq_context_segs, seq_imgs, seq_segs0, gt = data_lvrv_segmentation_propagation_mesa(mode = 'predict', fold = fold)
        
        seq_segs_pre_train = [sorted(seq_segs0[se][ind[se]:ind[se+1]] + seq_segs0[se+1][ind[se]:ind[se+1]], key=key_sort_files) for se in range(len(seq_segs0))[::2]]
        
        # seq_segs_gt_train = seq_segs1 #todo: test
        seq_segs_gt_train = [[a.replace('_predict_lvrv2_', '_crop_gt_', 1) for a in b ] for b in seq_segs_pre_train]
        seq_segs_gt_train = [[a.replace('_predict_lvrv_2D', '_crop_2D', 1) for a in b ] for b in seq_segs_gt_train]
        
        for_over = np.where(gt)[0]
        subjects = pd.DataFrame(excel_data, columns=['Subject']).to_numpy().flatten()[np.where(gt==1)[0]].tolist()
        subjects.append("All mean")
        subjects.append("All")
        print(subjects)
        
    elif dataset == 'mad_ous':
        
        dataset_name = 'MAD OUS'
        
        out_dir = config.out_dir_mad_ous
        info_file = os.path.join(out_dir, 'MAD_OUS_info.xlsx')
        excel_data = pd.read_excel(info_file)
        
        first = pd.DataFrame(excel_data, columns=['First']).to_numpy().flatten()
        last  = pd.DataFrame(excel_data, columns=['Last']).to_numpy().flatten()
        ind = [[first[i]] + [last[i]] for i in range(len(first))]
        ind = [item for sublist in ind for item in sublist]
        # ind = [first[i] + last[i] for i in range(len(first))]
        
        seq_context_imgs, seq_context_segs, seq_imgs, seq_segs0, gt = data_lvrv_segmentation_propagation_mad_ous(mode = 'predict', fold = fold)
        # seq_context_imgs, seq_context_segs, seq_imgs, seq_segs1, gt = data_mad_ous_lvrv_segmentation_propagation_acdc(mode = 'all', fold = fold)
        
        # seq_segs_pre_train = [item for sublist in seq_segs0 for item in sublist]
        seq_segs_pre_train = [sorted(seq_segs0[se][ind[se]:ind[se+1]] + seq_segs0[se+1][ind[se]:ind[se+1]], key=key_sort_files) for se in range(len(seq_segs0))[::2]]
        # seq_segs_pre_train = np.sort(seq_segs_pre_train)
        
        # seq_segs_gt_train = seq_segs1 #todo: test
        seq_segs_gt_train = [[a.replace('_predict_lvrv2_', '_crop_gt_', 1) for a in b ] for b in seq_segs_pre_train]
        seq_segs_gt_train = [[a.replace('_predict_lvrv_2D', '_crop_2D', 1) for a in b ] for b in seq_segs_gt_train]
        # seq_segs_gt_train = [[a.replace('predict_lvrv2_', 'crop_2D_gt_', 1) for a in b ] for b in seq_segs_pre_train] # name
        # seq_segs_gt_train = [[a.replace('MAD_OUS_predict_lvrv_2D', 'MAD_OUS_crop_2D', 1) for a in b ] for b in seq_segs_gt_train] # location
        
        for_over = np.where(gt)[0]
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
    
    dice_scores_train  = []
    dice_scores_train1 = []
    dice_scores_train2 = []
    dice_scores_train3 = []
    dice_scores_test   = []
    # print(f'Number of train subjects: {predict_sequence_train}. Number of test subjects: {predict_sequence_test}.')
    
    
    n_zero = 0
    n_segs = 0
    n_zero_3 = 0
    
    for i in for_over:    
        segs_pre = seq_segs_pre_train[i]
        #the segs_gt is not grouped 
        segs_gt = seq_segs_gt_train[i]
        
        if len(segs_pre) != len(segs_gt):
            print(f"Not same length! {i}")
        
        curr_dice   = []
        curr_dice_1 = []
        curr_dice_2 = []
        curr_dice_3 = []
        for frame in range(len(segs_pre)):
            
        
            segs_pre_im = np.array(iio.imread(segs_pre[frame]), dtype=float)
            segs_gt_im = np.array(iio.imread(segs_gt[frame]), dtype=float)
            
            # plt.plot(segs_pre_im[5])
            # plt.show()
            
            # print(f'{i}, {frame}: segs_pre_im: {K.sum(segs_pre_im)}') if K.sum(segs_pre_im) == 0 else False
            # print(f'{i}, {frame}: segs_gt_im: {K.sum(segs_gt_im)}') if K.sum(segs_gt_im) == 0 else False
            
            n_segs += 1
            if np.max(segs_pre_im) == 0:
                # print("Sum = 0.")
                # dice = np.nan
                n_zero += 1
            else:
                #skip the frames where we don't have a ground truth 
                if np.any(segs_gt_im) > 0: 
                
                # else:    
                    dice = dice_coef2(segs_gt_im, segs_pre_im, smooth=smooth).numpy()
                    
                    #split the data into the different regions
                    reg1_gt = np.where(segs_gt_im == 50, np.ones_like(segs_gt_im)*50, np.zeros_like(segs_gt_im))
                    reg2_gt = np.where(segs_gt_im == 100, np.ones_like(segs_gt_im)*100, np.zeros_like(segs_gt_im))
                    reg3_gt = np.where(segs_gt_im == 150, np.ones_like(segs_gt_im)*150, np.zeros_like(segs_gt_im))
                    reg1_pre = np.where(segs_pre_im == 50, np.ones_like(segs_pre_im)*50, np.zeros_like(segs_pre_im))
                    reg2_pre = np.where(segs_pre_im == 100, np.ones_like(segs_pre_im)*100, np.zeros_like(segs_pre_im))
                    reg3_pre = np.where(segs_pre_im == 150, np.ones_like(segs_pre_im)*150, np.zeros_like(segs_pre_im))
                    
                    dice1 = dice_coef2(reg1_gt, reg1_pre, smooth=smooth).numpy()
                    dice2 = dice_coef2(reg2_gt, reg2_pre, smooth=smooth).numpy()
                    dice3 = dice_coef2(reg3_gt, reg3_pre, smooth=smooth).numpy()
                
                    # if not np.isnan(dice):
                    curr_dice.append(dice)
                    curr_dice_1.append(dice1)
                    curr_dice_2.append(dice2)
                    #we'll only compare region 3 if it has been predicted
                    if np.any(reg3_pre) > 0:
                        curr_dice_3.append(dice3)
                
                if np.max(segs_pre_im) < 150:
                    n_zero_3 += 1
            
        dice_scores_train.append(curr_dice)
        dice_scores_train1.append(curr_dice_1)
        dice_scores_train2.append(curr_dice_2)
        dice_scores_train3.append(curr_dice_3)
        
    # for i in range(predict_sequence_test):    
    #     segs_pre = seq_segs_pre_test[i]
    #     #the segs_gt is not grouped 
    #     segs_gt = seq_segs_gt_test[i]
        
    #     if len(segs_pre) != len(segs_gt):
    #         print(f"Not same length! {i}")
        
    #     segs_pre_im = np.array([iio.imread(se) for se in segs_pre], dtype=float)
    #     segs_gt_im = np.array([iio.imread(se) for se in segs_gt], dtype=float)

    #     curr_dice = dice_coef2(segs_gt_im, segs_pre_im).numpy()
    #     dice_scores_test.append(curr_dice)
    
    # print(f'\nNumber of train subjects: {predict_sequence_train}.')
    final_str = f'Number of subjects: {predict_sequence_train}.\n'
    final_str += f'Number of failed segs: {n_zero} of {n_segs}, {int(100*n_zero/n_segs)}%.\n'
    #of the number of successful segmentations, how many failed to find region 3
    final_str += f'Number of failed region 3 segs: {n_zero_3} of {n_segs-n_zero}, {int(100*n_zero_3/(n_segs-n_zero))}%.\n\n'
    
    latex_tabel = ''
    # pd.options.display.float_format = "{:.3f}".format
    for i, list_n in enumerate([dice_scores_train, dice_scores_train1, dice_scores_train2, dice_scores_train3]):
        dice_flat = [item for sublist in list_n for item in sublist]
        dice_mean = [np.mean(sublist) for sublist in list_n]
        list_n.append(dice_mean)
        list_n.append(dice_flat)
        df = pd.DataFrame(list_n, index=subjects).transpose()
        print()
        if i == 0:
            region = "All regions"
        elif i == 1:
            region = "The left ventricle"
        elif i == 2:
            region = "The wall"
        elif i == 3:
            region = "The right ventricle"
        # else:
        #     region = f"Region {i}"
        print(f'{region}:')
        des = df.describe()
        des.loc['count'] = des.loc['count'].astype(int).astype(str)
        des.iloc[1:] = des.iloc[1:].applymap('{:.4f}'.format)
        pd.set_option('display.max_columns', 12)
        pd.set_option('display.width', 1000)
        print(des)
        
        text_file = open(out_dir + f"/dice_res_{dataset}_{i}.csv", "w")
        n = text_file.write(str(des) + final_str)
        text_file.close()
        print()
        
        if do_latex_table:
            des = des.loc[['count', 'mean', 'std', 'min', 'max']]
            
            if dataset == 'mesa':
                des.columns = des.columns.str.replace('MES', '')
                des.columns = des.columns.str.replace(r'01$', '', regex=True)
                des.columns = des.columns.str.replace(r'^00', '', regex=True)
            extra = ' "MES00" and "01" have been removed from the beginning and end of the subject names.' if (dataset == 'mesa') else ""
            latex_tabel += des.to_latex(bold_rows=True, label=f'tab:{dataset}_{i}', caption=f'Table of DICE-scores for the {dataset_name} dataset at {region.lower()}.{extra}', position='H', column_format='c'*(len(for_over)+3))
            latex_tabel += '\n'
    
    if do_latex_table:
        print('\n' + latex_tabel)    
        text_file = open(f"latexcode_dice_table_{dataset}.md", "wt")
        n = text_file.write(latex_tabel)
        text_file.close()
    
    
    print(final_str)
    
    return [dice_scores_train, dice_scores_train1, dice_scores_train2, dice_scores_train3], subjects #, dice_scores_test


if __name__ == '__main__':
    #I think dice_coef2 is the correct one
    # dice, subjects = calculate_dice_segmentation(smooth=.0, do_latex_table=False)
    dice, subjects = calculate_dice_segmentation('mad_ous', smooth=.0)
    dice, subjects = calculate_dice_segmentation('mesa', smooth=.0)
    
    
    
