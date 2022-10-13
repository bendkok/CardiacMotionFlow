""" A function to generate the lists of files for ApparentFlow-net """

import sys
sys.path.append('..')

import os
import math
import pandas as pd
import numpy as np
import re

import config

def key_sort_files(value):
    #from: https://stackoverflow.com/a/59175736/15147410
    """Extract numbers from string and return a tuple of the numeric values"""
    return tuple(map(int, re.findall('\d+', value)))

def flatten(l):
    return [item for sublist in l for item in sublist]

def data_mesa_apparentflow(mode='all', fold = 1, use_data_file=True):

    data_dir = "C:/Users/benda/Documents/Jobb_Simula/MAD_motion/MESA_crop_2D/{}" #config.acdc_data_dir
    code_dir = config.code_dir
    out_dir = config.out_dir
    
    if use_data_file:
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
        
        test_subjects = all_subjects
        
    else:    
        test_subjects = ['MES00{}01'.format(str(x).zfill(3)) for x in range(100)]
        all_subjects  = test_subjects #todo: change when labels are introduced 


    if mode == 'all':
        subjects = all_subjects
    elif mode in ['train', 'val']: #mode == 'train':
        subjects = [x for i,x in enumerate(all_subjects) if (i % 5) != (fold % 5)]
    elif mode in ['test', 'predict']:
        subjects = test_subjects
    else:
        print('Incorrect mode')

    print(subjects)

    excluded_slice_ratio = config.excluded_slice_ratio

    seq_instants = config.acdc_seq_instants


    # info_file = os.path.join(code_dir, 'acdc_info', 'acdc_info.txt')
    # with open(info_file) as in_file:
    #     subject_info = in_file.readlines()

    # subject_info = [x.strip() for x in subject_info]
    # subject_info = [ y.split()[0:2] + [float(z) for z in y.split()[2:]] for y in subject_info]

    #todo: change gt stuff
    gt_base_file = os.path.join(code_dir, 'acdc_info', 'acdc_gt_base.txt')

    with open(gt_base_file) as g_file:
        gt_base_info = g_file.readlines()

    gt_base_info = [x.strip() for x in gt_base_info]
    gt_base_info = [ [y.split()[0]] + [int(z) for z in y.split()[1:]] for y in gt_base_info]


    print('There will be {} used subjects'.format(len(subjects)) ) 

    img_list0 = []
    img_list1 = []
    seg_list0 = []
    seg_list1 = []

    segmented_pair_count = 0
    unsegmented_pair_count = 0
    for s, subject in enumerate(subjects):
        #print(subject)
        instants = instants_list[s] # int([x for x in subject_info if x[0] == subject][0][2])
        slices = slices_list[s] # int([x for x in subject_info if x[0] == subject][0][5])
        ed_instant = ed_list[s] # int([x for x in subject_info if x[0] == subject][0][3])
        es_instant = es_list[s] # int([x for x in subject_info if x[0] == subject][0][4])
        subject_dir = data_dir.format(subject)


        if mode in ['test', 'predict']:
            start_slice = 0
            end_slice = slices
        else:
            #todo: take a look at this, check if it works
            base_slice =  int([x for x in gt_base_info if x[0] == subject][0][1])
            apex_slice =  int([x for x in gt_base_info if x[0] == subject][0][2])
            es_base_slice =  int([x for x in gt_base_info if x[0] == subject][0][3])
            es_apex_slice =  int([x for x in gt_base_info if x[0] == subject][0][4])
        
            # The start_slice is smaller than the end_slice
            start_slice = base_slice + int(round((apex_slice + 1 - base_slice) * excluded_slice_ratio))
            end_slice = apex_slice + 1 - int(round((apex_slice + 1 - base_slice) * excluded_slice_ratio))
        
        img_names = sorted(os.listdir(subject_dir), key=key_sort_files)
        all_files = [os.path.join(subject_dir, file) for file in img_names]
        # for sl in range(slices):
        #     for t in range(instants):
        #         # img_path = os.path.join(subject_dir, os.listdir(subject_dir)[t+sl*instants])
        #         img_path = all_files[t+sl*instants]
        
        # print(len(all_files), ed_instant, instants, start_slice, end_slice)
        # print([ed_instant+i*instants for i in range(start_slice, end_slice)])
        # print(len([[all_files[ed_instant+i*instants]]*instants for i in range(start_slice, end_slice) ]))
        img_list0 += flatten([[all_files[ed_instant+i*instants]]*instants for i in range(start_slice, end_slice) ])
        img_list1 += all_files
        if len(img_list0) != len(img_list1):
            print("\nimg_list0 and img_list1 not same length for ", subject, "\n")

        # for i in range(start_slice, end_slice):    
        #     for t in range(0, instants):
                
        #     #     for sl in range(slices):
        #     # for t in range(instants):
        #         img_path = all_files[t+i*instants] #os.path.join(subject_dir, os.listdir(subject_dir)[t+i*instants])
                
        #         img_path0 = all_files[ed_instant+i*instants]
        #         img0 = os.path.join(subject_dir, 'crop_2D_{}_{}.png'.format(str(i).zfill(2), str(ed_instant).zfill(2)) )
        #         img1 = os.path.join(subject_dir, 'crop_2D_{}_{}.png'.format(str(i).zfill(2), str(t).zfill(2)) )
        #         #todo: look at seg stuff, uses gt
        #         """
        #         if t == es_instant:
        #             seg0 = os.path.join(subject_dir, 'crop_2D', 'crop_2D_gt_{}_{}.png'.format(str(i).zfill(2), str(ed_instant).zfill(2)) )
        #             seg1 = os.path.join(subject_dir, 'crop_2D', 'crop_2D_gt_{}_{}.png'.format(str(i).zfill(2), str(t).zfill(2)) )
        #             segmented_pair_count += 1
        #         else:
        #             seg0 = os.path.join(subject_dir, 'crop_2D', 'crop_2D_gt_{}_{}.png'.format(str(i).zfill(2), str(-1).zfill(2)) )
        #             seg1 = os.path.join(subject_dir, 'crop_2D', 'crop_2D_gt_{}_{}.png'.format(str(i).zfill(2), str(-1).zfill(2)) )
        #             unsegmented_pair_count += 1
        #         seg_list0.append(seg0)
        #         seg_list1.append(seg1)
        #         """
        #         img_list0.append(img0)
        #         img_list1.append(img1)
            

    print('pair count = {}'.format(len(img_list0)) )
    print('segmented_pair_count = {}'.format(segmented_pair_count), 'unsegmented_pair_count = {}'.format(unsegmented_pair_count))

    # return img_list0, img_list1, seg_list0, seg_list1
    return img_list0, img_list1, img_list0, img_list1


if __name__ == '__main__':
    
    data_mesa_apparentflow("predict")
