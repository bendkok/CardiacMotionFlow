# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:29:03 2022

@author: benda
"""

import matplotlib.pyplot as plt 
import cv2 as cv
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm
import sys


from segmentation.data_lvrv_segmentation_propagation_acdc import data_lvrv_segmentation_propagation_acdc
from segmentation.data_lvrv_segmentation_propagation_mesa import data_lvrv_segmentation_propagation_mesa
from segmentation.data_lvrv_segmentation_propagation_mad_ous import data_lvrv_segmentation_propagation_mad_ous

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


def key_sort_files(value):
    #from: https://stackoverflow.com/a/59175736/15147410
    """
    Extract numbers from string and return a tuple of the numeric values.
    Used to sort alpha-numerically.
    """
    return tuple(map(int, re.findall('\d+', value)))



def make_overlay_segmentation(dataset = 'mad_ous', display=False):
    
    fold = 0 #int(sys.argv[1])
    # print('fold = {}'.format(fold))
    # if fold == 0:
    #     mode = 'predict'
    # elif fold in range(1,6):
    #     mode = 'val_predict'
    # else:
    #     print('Incorrect fold')
        
        
    if dataset == 'acdc':
        
        seq_segs_pre_train = []
        for f in range(0,6):
            if f == 0:
                mode = 'predict'
            elif f in range(1,6):
                mode = 'val_predict'
            seq_context_imgs, seq_context_segs, seq_imgs, seq_segs = data_lvrv_segmentation_propagation_acdc(mode = mode, fold = f)
            seq_segs_pre_train += [seq_segs[se] + seq_segs[se+1] for se in range(len(seq_segs))[::2]]
        
        seq_segs_pre_train = np.sort(seq_segs_pre_train)
        
        seq_segs_gt_train = [[a.replace('predict_lvrv2_', 'crop_2D_gt_', 1) for a in b ] for b in seq_segs_pre_train]
        seq_segs_gt_train = [[a.replace('predict_2D', 'crop_2D', 1) for a in b ] for b in seq_segs_gt_train]
        
        seq_crop = [[a.replace('crop_2D_gt_', 'crop_2D_', 1) for a in b ] for b in seq_segs_gt_train]
        
        for_over = range(len(seq_segs_pre_train))
        subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(1, 151)]
        
    elif dataset == 'mesa':
        
        seq_context_imgs, seq_context_segs, seq_imgs, seq_segs0, gt = data_lvrv_segmentation_propagation_mesa(mode = 'predict', fold = fold)
        
        seq_segs_pre_train = [sorted(seq_segs0[se] + seq_segs0[se+1], key=key_sort_files) for se in range(len(seq_segs0))[::2]]
        seq_segs_gt_train = [[a.replace('_predict_lvrv2_', '_crop_gt_', 1) for a in b ] for b in seq_segs_pre_train]
        seq_segs_gt_train = [[a.replace('_predict_lvrv_2D', '_crop_2D', 1) for a in b ] for b in seq_segs_gt_train]
        seq_crop = [[a.replace('_crop_gt_', '_crop_', 1) for a in b ] for b in seq_segs_gt_train]
        
        for_over = np.where(gt)[0]
        
        out_dir = config.out_dir_mesa
        info_file = os.path.join(out_dir, 'MESA_info.xlsx')
        excel_data = pd.read_excel(info_file)
        # subjects0 = pd.DataFrame(excel_data, columns=['Subject']).to_numpy().flatten().tolist()
        subjects = pd.DataFrame(excel_data, columns=['Subject']).to_numpy().flatten()[np.where(gt==1)[0]].tolist()
        
    elif dataset == 'mad_ous':
        
        seq_context_imgs, seq_context_segs, seq_imgs, seq_segs0, gt = data_lvrv_segmentation_propagation_mad_ous(mode = 'predict', fold = fold)
        
        seq_segs_pre_train = [sorted(seq_segs0[se] + seq_segs0[se+1], key=key_sort_files) for se in range(len(seq_segs0))[::2]]
        seq_segs_gt_train = [[a.replace('_predict_lvrv2_', '_crop_gt_', 1) for a in b ] for b in seq_segs_pre_train]
        seq_segs_gt_train = [[a.replace('_predict_lvrv_2D', '_crop_2D', 1) for a in b ] for b in seq_segs_gt_train]
        seq_crop = [[a.replace('_crop_gt_', '_crop_', 1) for a in b ] for b in seq_segs_gt_train]
        
        for_over = np.where(gt)[0]        
        
        out_dir = config.out_dir_mad_ous
        info_file = os.path.join(out_dir, 'MAD_OUS_info.xlsx')
        excel_data = pd.read_excel(info_file)
        # subjects0 = pd.DataFrame(excel_data, columns=['Subject']).to_numpy().flatten().tolist()
        subjects = pd.DataFrame(excel_data, columns=['Subject']).to_numpy().flatten()[np.where(gt==1)[0]].tolist()
        
    else:
        print("Unkown dataset.")
        raise 

    # seq_segs = the ground thruths
    
    predict_sequence_train = len(seq_segs_pre_train)
    # predict_sequence_test = len(seq_segs_pre_test)
    
    # print(f'Number of train subjects: {predict_sequence_train}. Number of test subjects: {predict_sequence_test}.')
    print(f'\nNumber of subjects: {predict_sequence_train}.')
    
    
    for ii,i in enumerate(tqdm(for_over, file=sys.stdout)):
        with nostdout():

            print(f"Subject: {subjects[ii]}")
            segs_pre = seq_segs_pre_train[i]
            segs_crop = seq_crop[i]
            #the segs_gt is not grouped 
            segs_gt = seq_segs_gt_train[i]
            
            if len(segs_pre) != len(segs_gt):
                print(f"Not same length! {i}")
            
            for frame in range(len(segs_pre)):
                
                pre = cv.imread(segs_pre[frame], 0)
                gt  = cv.imread(segs_gt[frame], 0)
                # pre = np.where(pre==150, pre, np.zeros_like(pre))
                # gt  = np.where(gt ==150, gt,  np.zeros_like(gt) )
                
                #this is bit of a hack, but it makes the 'jet' color consistent
                if np.max(pre) > 1:
                    pre[0,0] = 150
                masked = np.ma.masked_where(pre == 0, pre)
                masked_gt = np.ma.masked_where(gt == 0, gt)
                crop = cv.imread(segs_crop[frame], 0)
                
                if dataset == 'acdc':
                    save_dest = segs_pre[frame].replace('predict_lvrv2', 'comp_lvrv2', 1)
                    save_dest = save_dest.replace('predict_2D', 'comp_2D', 1)
                    save_dest_gt = save_dest.replace('comp_lvrv2', 'gt_comp_lvrv2', 1)
                    # os.makedirs(re.sub('(.*)\\\\.*', '\g<1>/', save_dest), exist_ok=True)
                else:
                    save_dest = segs_pre[frame].replace('_predict_lvrv_2D', '_comp_lvrv_2D', 1)
                    save_dest = save_dest.replace('_predict_lvrv2_', '_comp_lvrv2_', 1)
                    save_dest_gt = save_dest.replace('_comp_lvrv2_', '_comp_lvrv2_gt_', 1)
                os.makedirs(re.sub('(.*)\\\\.*', '\g<1>/', save_dest), exist_ok=True)
                
                #creates an overlayed image for prediction
                plt.axis('off')
                plt.imshow(crop,'gray', interpolation='none')
                plt.imshow(masked, 'jet', interpolation='none', alpha=0.3)
                plt.savefig(save_dest, bbox_inches='tight', pad_inches=0)
                if display:
                    plt.show()                
                else:
                    plt.clf()
                
                # if '41_predict_lvrv2_183.0_triggertime_10.0' in segs_pre[frame]:
                #     print(41)
                    
                #crate overlay for gt
                try:
                    plt.axis('off')
                    plt.imshow(crop,'gray', interpolation='none')
                    plt.imshow(masked_gt, 'jet', interpolation='none', alpha=0.3)
                    plt.savefig(save_dest_gt, bbox_inches='tight', pad_inches=0)
                    if display:
                        plt.show()                
                    else:
                        plt.clf()
                except:
                    ""
                        
            
            
if __name__ == '__main__':
    
    # make_overlay_segmentation(dataset = 'mad_ous', display=True)
    make_overlay_segmentation(dataset = 'mesa',    display=False)
    # make_overlay_segmentation(dataset = 'acdc',    display=False)
    








