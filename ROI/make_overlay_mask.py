# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 12:29:57 2022

@author: benda
"""

import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
import cv2 as cv
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm
import sys
import csv


from ROI.data_roi_predict import data_roi_predict
from ROI.data_mesa_roi_predict import data_mesa_roi_predict
from ROI.data_mad_ous_roi_predict import data_mad_ous_roi_predict

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



def make_overlay_mask(dataset = 'mad_ous', display=False):
    
    fold = 0 #int(sys.argv[1])
    # print('fold = {}'.format(fold))
    # if fold == 0:
    #     mode = 'predict'
    # elif fold in range(1,6):
    #     mode = 'val_predict'
    # else:
    #     print('Incorrect fold')
        
        
    if dataset == 'acdc':
        predict_img_list, predict_gt_list = data_roi_predict()
        predict_mask_list = [file_name.replace('original_2D', 'mask_original_2D', 2) for file_name in predict_img_list]
        overlay_list = [file_name.replace('original_2D', 'mask_overlay', 2) for file_name in predict_img_list]
        
        path = './acdc_info'
        
    elif dataset == 'mesa':
        
        predict_img_list, predict_gt_list, subject_dir_list, original_2D_paths, gt = data_mesa_roi_predict()
        
        predict_mask_list = [file_name.replace('preprocess', 'mask', 1) for file_name in predict_img_list]
        predict_mask_list = [file_name.replace('sliceloc', 'mask', 1) for file_name in predict_mask_list ]
        
        overlay_list = [file_name.replace('sliceloc', 'mask_overlay', 1) for file_name in predict_img_list]
        overlay_list = [file_name.replace('preprocess_original_2D', 'mask_overlay', 1) for file_name in overlay_list]
        
    elif dataset == 'mad_ous':
        
        predict_img_list, predict_gt_list, subject_dir_list, original_2D_paths, gt = data_mad_ous_roi_predict()
        
        predict_mask_list = [file_name.replace('preprocess', 'mask', 1) for file_name in predict_img_list]
        predict_mask_list = [file_name.replace('sliceloc', 'mask', 1) for file_name in predict_mask_list ]
        
        overlay_list = [file_name.replace('sliceloc', 'mask_overlay', 1) for file_name in predict_img_list]
        overlay_list = [file_name.replace('preprocess_original_2D', 'mask_overlay', 1) for file_name in overlay_list]
        
    else:
        print("Unkown dataset.")
        raise 
        
    if dataset != 'acdc':
        try:
            with open(f"../../MAD_motion/{dataset.upper()}/cropped_area_{dataset}.csv", newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                areas = []
                for row in spamreader:
                    areas.append(np.array(row[0].split(','), dtype=int))
        except:
            with open(f"../MAD_motion/{dataset.upper()}/cropped_area_{dataset}.csv", newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                areas = []
                for row in spamreader:
                    areas.append(np.array(row[0].split(','), dtype=int))
        areas = np.array(areas)
        
        # print([subject_dir_list[0].replace('_sorted', '_preprocess_original_2D') in f for f in predict_img_list])
        b = []
        for i in range(len(subject_dir_list)):
            b += [i for s in predict_img_list if subject_dir_list[i].replace('_set1_sorted', '_preprocess_original_2D').replace('_sorted', '_preprocess_original_2D') in s]
    
    # print(b)
    print(f'\nNumber of masks: {len(predict_mask_list)}.')
    
    for i,file in enumerate(tqdm(predict_img_list, file=sys.stdout)):
        with nostdout():

            img_file = predict_img_list[i]
            mask_file = predict_mask_list[i]
            overlay_file = overlay_list[i]
            
            
            os.makedirs(os.path.dirname(overlay_file), exist_ok=True)
                
            img  = np.rot90(cv.imread(img_file,  0)[::-1, ::-1], 3)[::-1, :]
            mask = np.rot90(cv.imread(mask_file, 0)[::-1, ::-1], 3)[::-1, :]

            masked = np.ma.masked_where(mask == 0, mask)
            # crop_image_data
            #creates an overlayed image for prediction
            plt.axis('off')
            plt.imshow(img,'gray', interpolation='none')
            plt.imshow(masked, 'Purples_r', interpolation='none', alpha=0.35)
            # plt.plot([ar[0], ar[2]+1], [ar[0], ar[3]+1])
            if dataset != 'acdc':
                ar = areas[b[i]]
                box = [[ar[0], ar[2]+1], [ar[0], ar[3]+1], [ar[1], ar[3]+1], [ar[1], ar[2]+1], [ar[0], ar[2]+1]]
                # plt.gca().add_patch(Rectangle((ar[0], ar[2]+1),ar[1]-ar[0],ar[3]-ar[2],linewidth=1,edgecolor='orange',facecolor='none'))
                plt.plot(*zip(*box), color='orange')
            plt.savefig(overlay_file, bbox_inches='tight', pad_inches=0)
            if display:
                plt.show()                
            else:
                plt.clf()
                
            #need to choose colors:
            #BrBG, Blues_r, 
            #Purples_r, Greens_r, Reds_r, ocean, RdGy, plasma, terrain, copper_r, tab20, tab20_r
                
            """
            Possible color choices:
            'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 
            'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 
            'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 
            'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 
            'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 
            'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 
            'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 
            'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 
            'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 
            'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 
            'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 
            'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
            """
            
                        
            
            
if __name__ == '__main__':
    
    make_overlay_mask(dataset = 'mad_ous', display=True )
    make_overlay_mask(dataset = 'mesa',    display=True )
    make_overlay_mask(dataset = 'acdc',    display=False)
    








