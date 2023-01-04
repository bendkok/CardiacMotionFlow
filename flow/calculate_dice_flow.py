# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 00:12:06 2023

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
    y_true_f = tf.where(y_true > 0.5, K.ones_like(y_true), K.zeros_like(y_true))
    y_pred_f = tf.where(y_pred > 0.5, K.ones_like(y_pred), K.zeros_like(y_pred))
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def key_sort_files(value):
    #from: https://stackoverflow.com/a/59175736/15147410
    """
    Extract numbers from string and return a tuple of the numeric values.
    Used to sort alpha-numerically.
    """
    return tuple(map(int, re.findall('\d+', value)))


def calculate_dice_flow(dataset = 'acdc', smooth=1, do_latex_table=True):
    
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
        code_dir = config.code_dir
        
        seq_segs_pre_train = []
        seq_segs_gt_train  = []
        for f in range(1,6):
            seq_context_imgs, seq_context_segs, seq_imgs, seq_segs = data_lvrv_segmentation_propagation_acdc(mode = 'val_predict', fold = f)
            seq_segs_gt_train  += [seq_segs[se  ] for se in range(len(seq_segs))[::2]]
            seq_segs_pre_train += [seq_segs[se+1] for se in range(len(seq_segs))[::2]]
        
        seq_segs_pre_train = np.sort(seq_segs_pre_train)
        seq_segs_gt_train  = np.sort(seq_segs_gt_train)
        
        seq_segs_gt_train = [[a.replace('predict_lvrv2_', 'crop_2D_gt_', 1) for a in b ] for b in seq_segs_gt_train]
        seq_segs_gt_train = [[a.replace('predict_2D', 'crop_2D', 1) for a in b ] for b in seq_segs_gt_train]
        
        seq_segs_flow_train = [[a.replace('predict_lvrv2_', 'predict_flow_warp2_', 1) for a in b ] for b in seq_segs_pre_train]
        
        
        for_over = range(len(seq_segs_pre_train))
        subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(1, 101)]#.append("All")
        subjects.append("All mean")
        subjects.append("All")
        
        excluded_slice_ratio = config.excluded_slice_ratio
        
        gt_base_file = os.path.join(code_dir, 'acdc_info', 'acdc_gt_base.txt')

        with open(gt_base_file) as g_file:
            gt_base_info = g_file.readlines()
    
        gt_base_info = [x.strip() for x in gt_base_info]
        gt_base_info = [ [y.split()[0]] + [int(z) for z in y.split()[1:]] for y in gt_base_info]
        
        for s,subject in enumerate(subjects[:-2]):
            
            base_slice =  int([x for x in gt_base_info if x[0] == subject][0][1])
            apex_slice =  int([x for x in gt_base_info if x[0] == subject][0][2])
            
            start_slice = base_slice + int(round((apex_slice + 1 - base_slice) * excluded_slice_ratio))
            # start_slice = base_slice + int(round((apex_slice - base_slice) * excluded_slice_ratio))
            end_slice = apex_slice + 1 - int(round((apex_slice + 1 - base_slice) * excluded_slice_ratio))
            
            seq_segs_flow_train[s] = seq_segs_flow_train[s][start_slice:end_slice]
            seq_segs_gt_train[s]   = seq_segs_gt_train[s][start_slice:end_slice]
        
    
        
    elif dataset == 'mesa':

        dataset_name = 'MESA'
        
        out_dir = config.out_dir_mesa
        info_file = os.path.join(out_dir, 'MESA_info.xlsx')
        excel_data = pd.read_excel(info_file)
        
        seq_context_imgs, seq_context_segs, seq_imgs, seq_segs0, gt = data_lvrv_segmentation_propagation_mesa(mode = 'predict', fold = fold)
        
        seq_segs_pre_train = [sorted(seq_segs0[se] + seq_segs0[se+1], key=key_sort_files) for se in range(len(seq_segs0))[::2]]
        
        seq_segs_gt_train = [[a.replace('_predict_lvrv2_', '_crop_gt_', 1) for a in b ] for b in seq_segs_pre_train]
        seq_segs_gt_train = [[a.replace('_predict_lvrv_2D', '_crop_2D', 1) for a in b ] for b in seq_segs_gt_train]
        
        seq_segs_flow_train = [[a.replace('_predict_lvrv2_', '_predict_flow_warp2_', 1) for a in b ] for b in seq_segs_pre_train]
        seq_segs_flow_train = [[a.replace('_predict_lvrv_2D', '_predict_2D', 1) for a in b ] for b in seq_segs_flow_train]
        
        for_over = np.where(gt)[0]
        subjects = pd.DataFrame(excel_data, columns=['Subject']).to_numpy().flatten()[np.where(gt==1)[0]].tolist()
        subjects.append("All mean")
        subjects.append("All")
        # print(subjects)
        
    elif dataset == 'mad_ous':
        
        dataset_name = 'MAD OUS'
        
        out_dir = config.out_dir_mad_ous
        info_file = os.path.join(out_dir, 'MAD_OUS_info.xlsx')
        excel_data = pd.read_excel(info_file)
        
        seq_context_imgs, seq_context_segs, seq_imgs, seq_segs0, gt = data_lvrv_segmentation_propagation_mad_ous(mode = 'predict', fold = fold)
        
        seq_segs_pre_train = [sorted(seq_segs0[se] + seq_segs0[se+1], key=key_sort_files) for se in range(len(seq_segs0))[::2]]
        
        seq_segs_gt_train = [[a.replace('_predict_lvrv2_', '_crop_gt_', 1) for a in b ] for b in seq_segs_pre_train]
        seq_segs_gt_train = [[a.replace('_predict_lvrv_2D', '_crop_2D', 1) for a in b ] for b in seq_segs_gt_train]

        seq_segs_flow_train = [[a.replace('_predict_lvrv2_', '_predict_flow_warp2_', 1) for a in b ] for b in seq_segs_pre_train]
        seq_segs_flow_train = [[a.replace('_predict_lvrv_2D', '_predict_2D', 1) for a in b ] for b in seq_segs_flow_train]
        
        
        for_over = np.where(gt)[0]
        subjects = pd.DataFrame(excel_data, columns=['Subject']).to_numpy().flatten()[np.where(gt==1)[0]].tolist()
        subjects.append("All mean")
        subjects.append("All")
        # print(subjects)
        
        
    else:
        print("Unkown dataset.")
        raise 
        
    
    predict_sequence_train = len(seq_segs_pre_train)
    # predict_sequence_test = len(seq_segs_pre_test)
    
    dice_scores_train  = []
    dice_scores_train1 = []
    dice_scores_train2 = []
    dice_scores_train3 = []
    
    for i in for_over:    
        segs_flow = seq_segs_flow_train[i]
        #the segs_gt is not grouped 
        segs_gt = seq_segs_gt_train[i]
        
        if len(segs_flow) != len(segs_gt):
            print(f"Not same length! {i}")
        
        curr_dice   = []
        curr_dice_1 = []
        curr_dice_2 = []
        curr_dice_3 = []
        for frame in range(len(segs_flow)):
            
            segs_flow_im = np.array(iio.imread(segs_flow[frame]), dtype=float)
            segs_gt_im = np.array(iio.imread(segs_gt[frame]), dtype=float)
            
            
            #if there is no prediction
            if np.any(segs_flow_im) > 0 and np.any(segs_gt_im) > 0:
                #skip the frames where we don't have a ground truth 
                # if  True: #np.any(segs_gt_im) > 0: 
                
                # else:    
                dice = dice_coef2(segs_gt_im, segs_flow_im, smooth=smooth).numpy()
                
                #split the data into the different regions
                reg1_gt = np.where(segs_gt_im == 50, np.ones_like(segs_gt_im)*50, np.zeros_like(segs_gt_im))
                reg2_gt = np.where(segs_gt_im == 100, np.ones_like(segs_gt_im)*100, np.zeros_like(segs_gt_im))
                reg3_gt = np.where(segs_gt_im == 150, np.ones_like(segs_gt_im)*150, np.zeros_like(segs_gt_im))
                reg1_flow = np.where(segs_flow_im == 50, np.ones_like(segs_flow_im)*50, np.zeros_like(segs_flow_im))
                reg2_flow = np.where(segs_flow_im == 100, np.ones_like(segs_flow_im)*100, np.zeros_like(segs_flow_im))
                reg3_flow = np.where(segs_flow_im == 150, np.ones_like(segs_flow_im)*150, np.zeros_like(segs_flow_im))
                
                dice1 = dice_coef2(reg1_gt, reg1_flow, smooth=smooth).numpy()
                dice2 = dice_coef2(reg2_gt, reg2_flow, smooth=smooth).numpy()
                dice3 = dice_coef2(reg3_gt, reg3_flow, smooth=smooth).numpy()
            
                curr_dice.append(dice)
                curr_dice_1.append(dice1)
                curr_dice_2.append(dice2)
                #we'll only compare region 3 if it has been predicted
                if np.any(reg3_flow) > 0 and np.any(reg3_gt) > 0 :
                    curr_dice_3.append(dice3)
                
            
        dice_scores_train.append(curr_dice)
        dice_scores_train1.append(curr_dice_1)
        dice_scores_train2.append(curr_dice_2)
        dice_scores_train3.append(curr_dice_3)
        
        
    final_str = f'Number of subjects: {predict_sequence_train}.\n'
    
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
            region = "LVC" # "The left blood pool"
        elif i == 2:
            region = "LVM" #"The LV wall"
        elif i == 3:
            region = "RVC" #"The rigth blood pool"
        # else:
        #     region = f"Region {i}"
        print(f'{region}:')
        des = df.describe()
        des.loc['count'] = des.loc['count'].astype(int).astype(str)
        des.iloc[1:] = des.iloc[1:].applymap('{:.2f}'.format)
        pd.set_option('display.max_columns', 12)
        pd.set_option('display.width', 1000)
        print(des)
        
        # text_file = open(out_dir + f"/dice_res_{dataset}_{i}.csv", "w")
        # n = text_file.write(str(des) + final_str)
        # text_file.close()
        # print()
        
        if do_latex_table:
            des = des.loc[['count', 'mean', 'std', 'min', 'max']]
            
            if dataset == 'mesa':
                des.columns = des.columns.str.replace('MES', '')
                des.columns = des.columns.str.replace(r'01$', '', regex=True)
                des.columns = des.columns.str.replace(r'^00', '', regex=True)
            extra = ' "MES00" and "01" have been removed from the beginning and end of the subject names respectively.' if (dataset == 'mesa') else ""
            if np.isnan(np.array(des.values, dtype=float)).any():
                extra += ' The "nan" values comes if there are too few predictions.'
            latex_tabel += des.to_latex(bold_rows=True, label=f'tab:{dataset}_{i}', caption=f'Table of DICE-scores for the {dataset_name} dataset at {region.lower()}. "All mean" are the average of the mean values for each patient, while "All" are all the frames in total.{extra}', position='H', column_format='c'*(len(for_over)+3))
            latex_tabel += '\n'
    
    if do_latex_table:
        print('\n' + latex_tabel)    
        # text_file = open(f"latexcode_dice_table_{dataset}.md", "wt")
        # n = text_file.write(latex_tabel)
        # text_file.close()
    
    
    print(final_str)
    
    return [dice_scores_train, dice_scores_train1, dice_scores_train2, dice_scores_train3], subjects #, dice_scores_test


if __name__ == '__main__':
    #I think dice_coef2 is the correct one
    dice_acdc, subjects_acdc = calculate_dice_flow(smooth=.0, do_latex_table=False)
    dice_mad, subjects_mad   = calculate_dice_flow('mad_ous', smooth=.0, do_latex_table=False)
    dice_mesa, subjects_mesa = calculate_dice_flow('mesa', smooth=.0, do_latex_table=False)
    
    
    for i in range(len(dice_acdc)):
        dice_acdc[i] = [item for sublist in dice_acdc[i] for item in sublist]
        dice_mad[i]  = [item for sublist in dice_mad[i]  for item in sublist]
        dice_mesa[i] = [item for sublist in dice_mesa[i] for item in sublist]
    
    dice_df = pd.DataFrame([dice_mad[0], dice_mesa[0], dice_acdc[0]]).transpose()
    print(dice_df.describe())
    
    
    latex_table = """
\\begin{table}[H]
\\centering
\\caption{Table showing the mean (and standard deviation) of Dice-scores of the flow prediction ($L_{GT}$) for the different datasets, broken up by region. The bottom two are the values we found for ACDC and the values reported in Zheng et al. \\cite{zheng2019explainable}. }
\\bgroup
\\def\\arraystretch{1.4}%  1 is the default, change whatever you need
\\begin{tabular}{ccccc}
\\hline
\\textbf{Dataset}                   & \\textbf{Overall} & \\textbf{LVC} & \\textbf{LVM} & \\textbf{RVC} \\\\ \\hline\n"""
    latex_table += "MAD                                & {} & {} & {} & {} \\\\".format((*["{:.2f} ({:.2f})".format(np.mean(a), np.std(a)) for a in dice_mad])) + "\n"
    latex_table += "MESA                               & {} & {} & {} & {} \\\\".format((*["{:.2f} ({:.2f})".format(np.mean(a), np.std(a)) for a in dice_mesa])) + "\n"
    latex_table += "ACDC (found)                       & {} & {} & {} & {} \\\\".format((*["{:.2f} ({:.2f})".format(np.mean(a), np.std(a)) for a in dice_acdc])) + "\n"
    latex_table +="""ACDC (\\cite{zheng2019explainable}) & -           & 0.94 (0.07) & 0.84 (0.07) & 0.87 (0.19) \\\\ \\hline
\\end{tabular}
\\egroup
\\label{tab:dice:flow:all}
\\end{table}"""
    
    # formated_res = [f"{np.mean(a)} ({np.mean(a)})" for a in dice_mad] + [f"{np.mean(a)} ({np.mean(a)})" for a in dice_mesa] + [f"{np.mean(a)} ({np.mean(a)})" for a in dice_acdc]
    # latex_table = latex_table.format(formated_res) 
    
    print(latex_table)
    
    
    
    
    
    
    
    
    
