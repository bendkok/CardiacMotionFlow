# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:41:32 2022

@author: benda
"""

import sys
sys.path.append('..')

import os
import numpy as np
from PIL import Image
from scipy import interpolate
import nibabel as nib
import pandas as pd
import re
import pydicom
from PIL import Image as pil_image
import cv2 as cv


import config


def key_sort_files(value):
    #from: https://stackoverflow.com/a/59175736/15147410
    """Extract numbers from string and return a tuple of the numeric values"""
    return tuple(map(int, re.findall('\d+', value)))



def preprocess_mad_ous(dataset='mad_ous'):
    
    if dataset in ['mesa', 'MESA']:
        out_dir = config.out_dir_mesa
        info_file = os.path.join(out_dir, 'MESA_info.xlsx')
        excel_data = pd.read_excel(info_file)
    elif dataset in ['mad_ous', 'MAD_OUS']:
        out_dir = config.out_dir_mad_ous
        info_file = os.path.join(out_dir, 'MAD_OUS_info.xlsx')
        excel_data = pd.read_excel(info_file)
        
    
    data = pd.DataFrame(excel_data, columns=['Subject', 'Direcory', 'Filepath', 'ED', 'ES', 'Slices', 'Instants', 'GT'])
    all_subjects = data.Subject.to_numpy(dtype=str) #list of the subjects
    subject_dir_list = data.Direcory.to_numpy(dtype=str) #list of directory for each of the subjects
    original_2D_paths = data.Filepath.to_numpy(dtype=str) #list of directories where the files we use are
    
    instants_list = data.Instants.to_numpy(dtype=int)
    ed_list = data.ED.to_numpy(dtype=int)
    es_list = data.ES.to_numpy(dtype=int)
    slices_list = data.Slices.to_numpy(dtype=int)
    gt = data.GT.to_numpy(dtype=int)
        
    # data_dir = config.acdc_data_dir
    code_dir = config.code_dir
    
    

    # subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(1, 151)]

    print('There are {} subjects in total'.format(len(all_subjects)))
    
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # For each case
    for s,subject in enumerate(all_subjects):
        print('Processing {}'.format(subject) )

        # Define the paths
        subject_dir = subject_dir_list[s]
        subject_original_2D_dir = original_2D_paths[s]
        
        subject_preprocess_dir = os.path.join(out_dir, 'MAD_OUS_preprocess_original_2D', subject+'/').replace('\\', '/')
        os.makedirs(os.path.dirname(subject_preprocess_dir), exist_ok=True) #make sure the parent directory exists
        # if not os.path.exists(subject_preprocess_dir):
        #     os.makedirs(subject_preprocess_dir)

        # if not os.path.exists(subject_original_2D_dir):
        #     os.makedirs(subject_original_2D_dir)

        # sa_zip_file = os.path.join(subject_dir, '{}_4d.nii.gz'.format(subject))
        
        #a list with all the frames for this patient
        all_frames = sorted(os.listdir(subject_original_2D_dir), key=key_sort_files)
        
        instants = instants_list[s]
        slices = slices_list[s]
        
        subject_data0 = pydicom.read_file(os.path.join(subject_original_2D_dir, all_frames[0])).pixel_array.shape
        data_np = np.zeros((subject_data0[0], subject_data0[1], slices, instants))
        for t in range(instants):
            for sl in range(slices):
                data_np[:,:,sl,t] = pydicom.read_file(os.path.join(subject_original_2D_dir, all_frames[sl*instants+t])).pixel_array
                # img = nib.load(sa_zip_file)
                # data = img.get_data()
                # data_np = np.array(data)

        max_pixel_value = data_np.max()

        if max_pixel_value > 0:
            multiplier = 255.0 / max_pixel_value
        else:
            multiplier = 1.0

        print('max_pixel_value = {},  multiplier = {}'.format(max_pixel_value, multiplier))

        # rows = data.shape[0]
        # columns = data.shape[1]
        # slices = data.shape[2]
        # instants = data.shape[3]

        for t in range(instants):
            for sl in range(slices):
                s_t_image_file = os.path.join(subject_preprocess_dir, all_frames[sl*instants+t] )+'.png'
                # img = clahe.apply((data_np[:, :, sl, t]).astype('uint8'))
                # img = clahe.apply((data_np[:, ::-1, sl, t]).astype('uint8'))
                img = clahe.apply((np.rot90(data_np[:, ::-1, sl, t], 1) * multiplier).astype('uint8'))
                Image.fromarray(img).save(s_t_image_file)
                # Image.fromarray((data_np[:, ::-1, sl, t] * multiplier).astype('uint8')).save(s_t_image_file)

        # else:
        #     print('There is no SA image file for {}'.format(subject))


        



if __name__ == '__main__':
    preprocess_mad_ous()