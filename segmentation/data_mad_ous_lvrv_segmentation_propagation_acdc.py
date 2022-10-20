""" A function to generate the lists of files for LVRV-net inference """

import sys
sys.path.append('..')

import os
import math
import pandas as pd
import numpy as np
import re
import pathlib

import config

def key_sort_files(value):
    #from: https://stackoverflow.com/a/59175736/15147410
    """Extract numbers from string and return a tuple of the numeric values"""
    return tuple(map(int, re.findall('\d+', value)))

def flatten(l):
    return [item for sublist in l for item in sublist]

def data_mad_ous_lvrv_segmentation_propagation_acdc(mode='all', fold = 1, use_data_file=True):

    data_dir = "C:/Users/benda/Documents/Jobb_Simula/MAD_motion/MAD_OUS/MAD_OUS_crop_2D/{}" #config.acdc_data_dir
    code_dir = config.code_dir
    out_dir = config.out_dir_mad_ous
    
    if use_data_file:
        info_file = os.path.join(out_dir, 'MAD_OUS_info.xlsx') #todo: change to be function-input, with None as default
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



    # info_file = os.path.join(code_dir, 'acdc_info', 'acdc_info.txt')
    # with open(info_file) as in_file:
    #     subject_info = in_file.readlines()

    # subject_info = [x.strip() for x in subject_info]
    # subject_info = [ y.split()[0:2] + [float(z) for z in y.split()[2:]] for y in subject_info]


    print('There will be {} used subjects'.format(len(subjects)) ) 


    seq_context_imgs = []
    seq_context_segs = []
    seq_imgs = []
    seq_segs = []

    seq_context_imgs_no_group = []
    seq_context_segs_no_group = []
    seq_imgs_no_group = []
    seq_segs_no_group = []


    for s, subject in enumerate(subjects):
        #print(subject)
        instants = instants_list[s] # int([x for x in subject_info if x[0] == subject][0][2])
        slices = slices_list[s] # int([x for x in subject_info if x[0] == subject][0][5])
        ed_instant = ed_list[s] # int([x for x in subject_info if x[0] == subject][0][3])
        es_instant = es_list[s] # int([x for x in subject_info if x[0] == subject][0][4])
        subject_dir = data_dir.format(subject)

        start_slice = 0
        end_slice = slices 
        
        pred_dir = subject_dir.replace('MAD_OUS_crop_2D', 'MAD_OUS_predict_2D')
        os.makedirs(pred_dir, exist_ok=True)
        
        files = sorted(os.listdir(subject_dir), key=key_sort_files)

        for t in [ed_instant, es_instant]:
            # context_imgs = []
            # context_segs = []
            # imgs = []
            # segs = []
            
            context_img = ["dummy.png"] + files[t::instants][:-1]
            context_img = [os.path.join(subject_dir, i) for i in context_img]
            if mode in ['all', 'train', 'val']:    
                context_seg = [""]*instants #todo: add later
            elif mode in ['predict', 'val_predict']:
                context_seg = [i.replace("MAD_OUS_crop_2D", "MAD_OUS_predict_2D") for i in context_img]
                context_seg = [i.replace("_crop_", "_predict_lvrv2_") for i in context_seg]
            img = files[t::instants]
            img = [os.path.join(subject_dir, i) for i in img]
            if mode in ['all', 'train', 'val']:    
                seg = [""]*instants #todo: add later
            elif mode in ['predict', 'val_predict']:
                seg = [i.replace("MAD_OUS_crop_2D", "MAD_OUS_predict_lvrv_2D") for i in img]
                seg = [i.replace("_crop_", "_predict_lvrv2_") for i in seg]
            
            seq_context_imgs_no_group += context_img
            seq_context_segs_no_group += context_seg
            seq_imgs_no_group += img
            seq_segs_no_group += seg
            
            seq_context_imgs.append(context_img)
            seq_context_segs.append(context_seg)
            seq_imgs.append(img)
            seq_segs.append(seg)
            
            # for i in range(slices):
            # # for i in range(start_slice, end_slice):
            #     if i == start_slice: #why is this done?
            #         i_minus = -1
            #     else:
            #         i_minus = i - 1

                 
            #     context_img = os.path.join(subject_dir, 'crop_2D', 'crop_2D_{}_{}.png'.format(str(i_minus).zfill(2), str(t).zfill(2)) )
            #     if mode in ['all', 'train', 'val']:    
            #         context_seg = os.path.join(subject_dir, 'crop_2D', 'crop_2D_gt_{}_{}.png'.format(str(i_minus).zfill(2), str(t).zfill(2)) )
            #     elif mode in ['predict', 'val_predict']:
            #         context_seg = os.path.join(subject_dir, 'predict_2D', 'predict_lvrv2_{}_{}.png'.format(str(i_minus).zfill(2), str(t).zfill(2)) )



            #     img = os.path.join(subject_dir, 'crop_2D', 'crop_2D_{}_{}.png'.format(str(i).zfill(2), str(t).zfill(2)) )
            #     if mode in ['all', 'train', 'val']:
            #         seg = os.path.join(subject_dir, 'crop_2D', 'crop_2D_gt_{}_{}.png'.format(str(i).zfill(2), str(t).zfill(2)) )
            #     elif mode in ['predict', 'val_predict']:
            #         seg = os.path.join(subject_dir, 'predict_2D', 'predict_lvrv2_{}_{}.png'.format(str(i).zfill(2), str(t).zfill(2)) )

            #     seq_context_imgs_no_group.append(context_img)
            #     seq_context_segs_no_group.append(context_seg)
            #     seq_imgs_no_group.append(img)
            #     seq_segs_no_group.append(seg)


            #     context_imgs.append(context_img)
            #     context_segs.append(context_seg)
            #     imgs.append(img)
            #     segs.append(seg)

            
                

    if mode in ['all', 'train', 'val']:
        return seq_context_imgs_no_group, seq_context_segs_no_group, seq_imgs_no_group, seq_segs_no_group
    elif mode in ['predict', 'val_predict']:
        return seq_context_imgs, seq_context_segs, seq_imgs, seq_segs
        

if __name__ == '__main__':
    
    res = data_mad_ous_lvrv_segmentation_propagation_acdc("predict")
    # print(len(res[0]), len(res[0][0]))
    # print(res[0][0])

