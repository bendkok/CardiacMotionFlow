""" A function to generate the lists of files for ROI-net inference"""

import sys
sys.path.append('..')

import os
import pydicom
import pandas as pd
import numpy as np
import shutil as shu
import config
import re

def key_sort_files(value):
    #from: https://stackoverflow.com/a/59175736/15147410
    """Extract numbers from string and return a tuple of the numeric values"""
    return tuple(map(int, re.findall('\d+', value)))



def data_mesa_roi_predict(use_info_file=True, delete=False):
    data_dir = "C:/Users/benda/Documents/Jobb_Simula/MAD_motion/MESA/MESA_set1_sorted/{}" #config.acdc_data_dir
    # out_dir = "C:/Users/benda/Documents/Jobb_Simula/MAD_motion/" 
    out_dir = config.out_dir_mesa

    
    predict_img_list = []
    predict_gt_list = []
    subject_dir_list = []
    skipped = 0
    
    if delete:
        #clears the output folder incase any stray/old files are there
        if os.path.exists(os.path.join(out_dir, 'MESA_mask_original_2D')):            
            shu.rmtree(os.path.join(out_dir, 'MESA_mask_original_2D'))
    
    #we can either use a file with info on the different paitients, or we can auto-parse the directory
    if not use_info_file:        
        
        all_subjects = ['MES00{}01'.format(str(x).zfill(3)) for x in range(100)]
        ed_list = np.zeros(len(all_subjects), dtype=int)
        original_2D_paths = []
        
    else:
        info_file = os.path.join(out_dir, 'MESA_info.xlsx') #todo: change to be function-input, with None as default
        excel_data = pd.read_excel(info_file)
        data = pd.DataFrame(excel_data, columns=['Subject', 'Direcory', 'Filepath', 'ED', 'ES', 'Slices', 'Instants'])
        
        all_subjects = data.Subject.to_numpy(dtype=str) #list of the subjects
        subject_dir_list = data.Direcory.to_numpy(dtype=str) #list of directory for each of the subjects
        original_2D_paths = data.Filepath.to_numpy(dtype=str) #list of directories where the files we use are
        
        instants_list = data.Instants.to_numpy(dtype=int)
        ed_list = data.ED.to_numpy(dtype=int)
        es_list = data.ES.to_numpy(dtype=int)
        slices_list = data.Slices.to_numpy(dtype=int)
                 
    for s,subject in enumerate(all_subjects):
        
        if not use_info_file:
            
            subject_dir = data_dir.format(subject)
            subject_dir_list.append(subject_dir)
      
            #there were at least five different naming conventions and some patient don't have a short-axis scan
            if os.path.exists(os.path.join(subject_dir, 'short_axis_cine_(multi-slice)') ):
                original_2D_path = os.path.join(subject_dir, 'short_axis_cine_(multi-slice)')
            elif os.path.exists(os.path.join(subject_dir, 'short_axis_(10_slices)') ):
                original_2D_path = os.path.join(subject_dir, 'short_axis_(10_slices)')
            elif os.path.exists(os.path.join(subject_dir, 'short_axis_cine_(multi-slice') ):
                original_2D_path = os.path.join(subject_dir, 'short_axis_cine_(multi-slice')
            elif os.path.exists(os.path.join(subject_dir, 'short_axis_tag_90') ):
                original_2D_path = os.path.join(subject_dir, 'short_axis_tag_90')
            elif os.path.exists(os.path.join(subject_dir, 'derived-primary-other/short_axis_(10_slices)') ):
                original_2D_path = os.path.join(subject_dir, 'derived-primary-other/short_axis_(10_slices)')
            else: 
                print("Warning! No short-axis scan found for subject {}.".format(subject))
                skipped += 1
                continue
    
            #some patients have mulitple series, finds the one with the most slices
            if os.path.exists(os.path.join(original_2D_path, 'series_0')):
                subdirs = os.listdir(original_2D_path)
                longest = len(os.listdir(os.path.join(original_2D_path, subdirs[0])))
                longest_dir = subdirs[0]
                for sd in subdirs[1:]:
                    curr = len(os.listdir(os.path.join(original_2D_path, sd)))
                    if curr > longest:
                        longest = curr
                        longest_dir = sd
                original_2D_path = os.path.join(original_2D_path, longest_dir)        
            
            original_2D_paths.append(original_2D_path)
            
        else:
            original_2D_path = original_2D_paths[s]
            slices = slices_list[s]
            instants = instants_list[s]
            
              
        #where we'll store the masks
        subject_predict_dir = os.path.join(out_dir, 'MESA_mask_original_2D', subject)
        if not os.path.exists(subject_predict_dir):
            os.makedirs(subject_predict_dir)
        
        
            
        #a list with all the frames for this patient
        all_frames = sorted(os.listdir(original_2D_path), key=key_sort_files)
        
        
        if not use_info_file:
            #it seems like each patient has 20 frames for each slice, hope that's consistent
            #this way we don't need to use re or similar to get the desired files
            subject_data = pydicom.read_file(os.path.join(original_2D_path, all_frames[0]))
            instants = subject_data.CardiacNumberOfImages
            slices = int(len(all_frames)/instants)
            # if instants != 20:    
            #     print(subject, instants, slices)
    
        # Prediction on the ED stacks only
        used_instants = [ed_list[s]]
        #Here they tried to get only the data for ED. MESA dosen't record when that is, so I've just used 0 instead
        
        # for s in range(slices):
        for sl in range(int(round(slices * 0.1 + 0.001)), int(round(slices * 0.5 + 0.001))): #they use this long thing to decide which slices to use, not sure why
            for t in used_instants: #there's only one element here, but that might change in the future
                # print('Numbers: {:3.0f} {:2.0f} {:2.0f}'.format(t+sl*instants, t, sl))
                s_t_image_file = os.path.join(original_2D_path, all_frames[sl*instants+t]) #select the desired frame
                predict_img_list.append(s_t_image_file)
        # print(predict_img_list)
        # exit()
        
        
    
    predict_gt_list = ['']*len(predict_img_list) #this needs to be included
    print('\npredict_image_count = {}. '.format(len(predict_img_list)) )
    print(f"Skipped patients = {skipped}.\n")
    # print(predict_img_list)
    # print(predict_gt_list)
    # print(subject_dir_list)

    return predict_img_list, predict_gt_list, subject_dir_list, original_2D_paths



if __name__ == '__main__':
    # out = data_mesa_roi_predict(False)
    # print(out)
    # for o in out[0]:
    #     print(o)
        
    out = data_mesa_roi_predict(True)
    # # print(out)
    # for o in out[0]:
    #     print(o)


