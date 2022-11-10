""" The main file to launch ROI cropping according to the prediction of ROI-net """

import os
import sys
sys.path.append('..')

import numpy as np
import scipy
from scipy import ndimage
import math
import nibabel as nib
from PIL import Image
import pandas as pd

import multiprocessing.pool
from functools import partial

import pydicom
import re
from tqdm import tqdm

import config

from ROI.data_mesa_roi_predict import data_mesa_roi_predict
from ROI.data_mad_ous_roi_predict import data_mad_ous_roi_predict

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

# Auxiliary function
def determine_rectangle_roi(img_path):
    img = Image.open(img_path)
    columns, rows = img.size
    roi_c_min = columns
    roi_c_max = -1
    roi_r_min = rows
    roi_r_max = -1
    box = img.getbbox()
    if box:
        roi_r_min = box[0]
        roi_c_min = box[1]
        roi_r_max = box[2] - 1
        roi_c_max = box[3] - 1
    return [roi_c_min, roi_c_max, roi_r_min, roi_r_max]

# Auxiliary function
def determine_rectangle_roi2(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)
    connected_components, num_connected_components = ndimage.label(img_array)
    if (num_connected_components > 1):
        unique, counts = np.unique(connected_components, return_counts=True)
        max_idx = np.where(counts == max(counts[1:]))[0][0]
        single_component = connected_components * (connected_components == max_idx)
        img = Image.fromarray(single_component)

    columns, rows = img.size
    roi_c_min = columns
    roi_c_max = -1
    roi_r_min = rows
    roi_r_max = -1
    box = img.getbbox()
    if box:
        roi_r_min = box[0]
        roi_c_min = box[1]
        roi_r_max = box[2] - 1
        roi_c_max = box[3] - 1
    return [roi_c_min, roi_c_max, roi_r_min, roi_r_max]


def change_array_values(array):
    output = array
    for u in range(output.shape[0]):
        for v in range(output.shape[1]):
            if output[u,v] == 1:
                output[u,v] = 3
            elif output[u,v] == 3:
                output[u,v] = 1
    return output

def key_sort_files(value):
    #from: https://stackoverflow.com/a/59175736/15147410
    """Extract numbers from string and return a tuple of the numeric values"""
    return tuple(map(int, re.findall('\d+', value)))


def crop_according_to_roi(dataset='acdc', use_info_file=True):
    # The ratio that determines the width of the margin
    pixel_margin_ratio = 0.3
    
    # If for a case there is non-zero pixels on the border of ROI, the case is stored in
    # this list for further examination. This list is eventually empty for UK Biobank cases.
    # border_problem_subject = []

    
    if dataset == 'acdc':
    
        data_dir = config.acdc_data_dir
        code_dir = config.code_dir
    
        dilated_subjects = config.acdc_dilated_subjects
        hypertrophic_subjects = config.acdc_hypertrophic_subjects
        infarct_subjects = config.acdc_infarct_subjects 
        normal_subjects = config.acdc_normal_subjects
        rv_subjects = config.acdc_rv_subjects
        test_subjects = config.acdc_test_subjects
    
        train_subjects = dilated_subjects + hypertrophic_subjects + infarct_subjects + normal_subjects + rv_subjects
        all_subjects = train_subjects + test_subjects
    
        info_file = os.path.join(code_dir, 'acdc_info', 'acdc_info.txt')
        with open(info_file) as in_file:
            subject_info = in_file.readlines()
    
        subject_info = [x.strip() for x in subject_info]
        subject_info = [ y.split()[0:2] + [float(z) for z in y.split()[2:]] for y in subject_info]
        
    elif dataset in ['mesa', 'mad_ous']:
        # data_dir = "C:\\Users\\benda\\Documents\\Jobb_Simula\\MAD_motion\\MESA_set1_sorted\\{}" #config.acdc_data_dir
        if dataset == 'mesa':
            out_dir = config.out_dir_mesa
            info_file = os.path.join(out_dir, 'MESA_info.xlsx')
            predict_img_list, predict_gt_list, subject_dir_list, original_2D_paths = data_mesa_roi_predict(use_info_file, delete=False)
        elif dataset == 'mad_ous':
            out_dir = config.out_dir_mad_ous
            info_file = os.path.join(out_dir, 'MAD_OUS_info.xlsx')
            predict_img_list, predict_gt_list, subject_dir_list, original_2D_paths, has_gt = data_mad_ous_roi_predict(use_info_file, delete=False)
            
        # code_dir = config.code_dir
 
        #we have 100 subject so far, for now I'm setting all of them to be training set
        # train_subjects = ['MES00{}01'.format(str(x).zfill(3)) for x in range(100)] # dilated_subjects + hypertrophic_subjects + infarct_subjects + normal_subjects + rv_subjects
    
        # all_subjects = train_subjects #+ test_subjects
    
        
        excel_data = pd.read_excel(info_file)
        data = pd.DataFrame(excel_data, columns=['Subject', 'Direcory', 'Filepath', 'ED', 'ES', 'Slices', 'Instants'])
    
        
        all_subjects = data.Subject.to_numpy(dtype=str) #list of the subjects
        train_subjects = all_subjects[np.where(has_gt==1)[0]] # todo: change
        
        # subject_dir_list = data.Direcory.to_numpy(dtype=str) #list of directory for each of the subjects
        original_2D_paths = data.Filepath.to_numpy(dtype=str) #list of directories where the files we use are
        
        instants_list = data.Instants.to_numpy(dtype=int)
        ed_list = data.ED.to_numpy(dtype=int)
        es_list = data.ES.to_numpy(dtype=int)
        slices_list = data.Slices.to_numpy(dtype=int)
    else:
        print("Unknown dataset.")
        raise

    if dataset == 'acdc':
        for subject in tqdm(all_subjects, file=sys.stdout):
            with nostdout():
                print(subject)
                subject_dir = data_dir.format(subject)
                subject_mask_original_dir = os.path.join(subject_dir, 'mask_original_2D')
                crop_2D_path = os.path.join(subject_dir, 'crop_2D')
                if not os.path.exists(crop_2D_path):
                    os.makedirs(crop_2D_path)
                
                
                instants = int([x for x in subject_info if x[0] == subject][0][2])
                ed_instant = int([x for x in subject_info if x[0] == subject][0][3])
                es_instant = int([x for x in subject_info if x[0] == subject][0][4])
                slices = int([x for x in subject_info if x[0] == subject][0][5])
    
                used_instants_roi = [ed_instant]
                img_path_list = []
                for t in used_instants_roi:
                    for s in range(int(round(slices * 0.1 + 0.001)), int(round(slices * 0.5 + 0.001))):
                        s_t_mask_image_file = os.path.join(subject_mask_original_dir, 'mask_original_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                        img_path_list.append(s_t_mask_image_file)
    
    
            
                # Multithread
                pool = multiprocessing.pool.ThreadPool()
                function_partial = partial(determine_rectangle_roi2)
                roi_results = pool.map(function_partial, (img_path for img_path in img_path_list))
                roi_c_min = min([res[0] for res in roi_results])
                roi_c_max = max([res[1] for res in roi_results])
                roi_r_min = min([res[2] for res in roi_results])
                roi_r_max = max([res[3] for res in roi_results])
                pool.close()
                pool.join()
    
                # ROI size (without adding margin)
                roi_c_length = roi_c_max - roi_c_min + 1
                roi_r_length = roi_r_max - roi_r_min + 1
                roi_length = max(roi_c_length, roi_r_length)
                print('roi_length = {}'.format(roi_length) )
    
                written = '{0} {1} {2} {3} {4} {5}\n'.format(subject, roi_c_min, roi_c_max, roi_r_min, roi_r_max, roi_length)
                
    
                # The size of margin, determined by the ratio we defined above
                pixel_margin = int(round(pixel_margin_ratio * roi_length + 0.001))
    
                crop_c_min = ((roi_c_min + roi_c_max) // 2) - (roi_length // 2) - pixel_margin
                crop_c_max = crop_c_min + pixel_margin + roi_length - 1 + pixel_margin
                crop_r_min = ((roi_r_min + roi_r_max) // 2) - (roi_length // 2) - pixel_margin
                crop_r_max = crop_r_min + pixel_margin + roi_length - 1 + pixel_margin
    
    
                # Crop the original images
                image_file = os.path.join(subject_dir, '{}_4d.nii.gz'.format(subject))
                image_load = nib.load(image_file)
                image_data = image_load.get_data()
                original_r_min = max(0, crop_r_min)
                original_r_max = min(image_data.shape[0]-1, crop_r_max)
                original_c_min = max(0, crop_c_min)
                original_c_max = min(image_data.shape[1]-1, crop_c_max)
                crop_image_data = np.zeros((roi_length + 2 * pixel_margin, roi_length + 2 * pixel_margin,
                                            image_data.shape[2], image_data.shape[3]))
                crop_image_data[(original_r_min - crop_r_min):(original_r_max - crop_r_min + 1), 
                                (original_c_min - crop_c_min):(original_c_max - crop_c_min + 1), 
                                :, 
                                :] = \
                    image_data[original_r_min:(original_r_max + 1), 
                            original_c_min:(original_c_max + 1), 
                            :, 
                            :]
                crop_image_data = crop_image_data[::-1, ::-1, :, :]
                crop_image_file = os.path.join(subject_dir, 'crop_{}_4d.nii.gz'.format(subject))
                nib.save(nib.Nifti1Image(crop_image_data, np.eye(4)), crop_image_file)
    
                # Crop the original labels
                if subject in train_subjects:
                    for i in [ed_instant+1, es_instant+1]:
                        label_file = os.path.join(subject_dir, '{}_frame{}_gt.nii.gz'.format(subject,str(i).zfill(2)))
                        label_load = nib.load(label_file)
                        label_data = label_load.get_data()
                        crop_label_data = np.zeros((roi_length + 2 * pixel_margin, 
                            roi_length + 2 * pixel_margin,
                            image_data.shape[2]))
                        crop_label_data[(original_r_min - crop_r_min):(original_r_max - crop_r_min + 1), 
                                (original_c_min - crop_c_min):(original_c_max - crop_c_min + 1), 
                                :] = \
                            label_data[original_r_min:(original_r_max + 1), 
                            original_c_min:(original_c_max + 1), 
                            :]
                        crop_label_data = crop_label_data[::-1, ::-1, :]
                        crop_label_file = os.path.join(subject_dir,
                            'crop_{}_frame{}_gt.nii.gz'.format(subject,str(i).zfill(2)))
                        nib.save(nib.Nifti1Image(crop_label_data, np.eye(4)), crop_label_file)
    
    
            
                # Save cropped 2D images
                crop_image_data = nib.load(crop_image_file).get_data()
    
                max_pixel_value = crop_image_data.max()
    
                if max_pixel_value > 0:
                    multiplier = 255.0 / max_pixel_value
                else:
                    multiplier = 1.0
    
                print('max_pixel_value = {}, multiplier = {}'.format(max_pixel_value, multiplier) )
    
                for s in range(slices):
                    for t in range(instants):
                        s_t_image_file = os.path.join(crop_2D_path, 'crop_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                        Image.fromarray((np.rot90(crop_image_data[:, ::-1, s, t], 3) * multiplier).astype('uint8')).save(s_t_image_file)
    
    
                # Save cropped 2D labels
                if subject in train_subjects:
                    for s in range(slices):
                        for t in [ed_instant, es_instant]:
                            crop_label_file = os.path.join(subject_dir, 
                                'crop_{}_frame{}_gt.nii.gz'.format(subject,str(t+1).zfill(2)))
                            crop_label_data = nib.load(crop_label_file).get_data()
                            s_t_label_file = os.path.join(crop_2D_path, 'crop_2D_gt_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                            Image.fromarray((np.rot90(change_array_values(crop_label_data[:, ::-1, s]), 3) * 50).astype('uint8')).save(s_t_label_file)


    elif dataset in ['mesa', 'mad_ous']: #todo: make this cleaner
        for s,subject in enumerate(tqdm(all_subjects, file=sys.stdout)):
            with nostdout():
                print(f"Subject: {subject}")
                subject_dir = original_2D_paths[s]
                # subject_dir_frames = os.listdir(subject_dir)
                if dataset == 'mesa':
                    dataset_name = 'MESA'
                    sub_key = "MESA_set1_sorted/(MES0\d{6}).*/([0-9]{1,3}_)(sliceloc.*)"
                    sub_rep = 'MESA_crop_2D/\g<1>/\g<2>crop_\g<3>'
                elif dataset == 'mad_ous':
                    dataset_name = 'MAD_OUS'
                    sub_key = "MAD_OUS_sorted/([0-9]+)/cine.+/([0-9]{1,3}_)_sliceloc_(.*)"
                    sub_rep = 'MAD_OUS_crop_2D/\g<1>/\g<2>crop_\g<3>'
                
                subject_mask_original_dir = os.path.join(out_dir, f'{dataset_name}_mask_original_2D', subject)
                crop_2D_path = os.path.join(out_dir, f'{dataset_name}_crop_2D', subject)
                if not os.path.exists(crop_2D_path):
                    os.makedirs(crop_2D_path)
                
                
                instants = instants_list[s] # subject_data.CardiacNumberOfImages #int([x for x in subject_info if x[0] == subject][0][2])
                #for now I've just set ed/es to be the first and last frames. Don't think that's the best option
                ed_instant = ed_list[s] # 0 
                es_instant = es_list[s] # len(subject_dir_frames) -1 
                slices = slices_list[s] # int(len(subject_dir_frames)/20) #int([x for x in subject_info if x[0] == subject][0][5])
        
        
                img_path_list = os.listdir(subject_mask_original_dir)[::2]
                img_path_list = [os.path.join(subject_mask_original_dir, img) for img in img_path_list]
                
                # for t in used_instants_roi:
                #     for s in range(int(round(slices * 0.1 + 0.001)), int(round(slices * 0.5 + 0.001))):
                #         s_t_mask_image_file = os.path.join(subject_mask_original_dir, 'mask_original_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                #         img_path_list.append(s_t_mask_image_file)
        
            
                # Multithread
                pool = multiprocessing.pool.ThreadPool()
                function_partial = partial(determine_rectangle_roi2)
                roi_results = pool.map(function_partial, (img_path for img_path in img_path_list))
                roi_c_min = min([res[0] for res in roi_results])
                roi_c_max = max([res[1] for res in roi_results])
                roi_r_min = min([res[2] for res in roi_results])
                roi_r_max = max([res[3] for res in roi_results])
                pool.close()
                pool.join()
        
                # ROI size (without adding margin)
                roi_c_length = roi_c_max - roi_c_min + 1
                roi_r_length = roi_r_max - roi_r_min + 1
                roi_length = max(roi_c_length, roi_r_length)
                print('roi_length = {}'.format(roi_length) )
        
                written = '{0} {1} {2} {3} {4} {5}\n'.format(subject, roi_c_min, roi_c_max, roi_r_min, roi_r_max, roi_length)
                # print('Written: ' + written)
        
                # The size of margin, determined by the ratio we defined above
                pixel_margin = int(round(pixel_margin_ratio * roi_length + 0.001))
        
                crop_c_min = ((roi_c_min + roi_c_max) // 2) - (roi_length // 2) - pixel_margin
                crop_c_max = crop_c_min + pixel_margin + roi_length - 1 + pixel_margin
                crop_r_min = ((roi_r_min + roi_r_max) // 2) - (roi_length // 2) - pixel_margin
                crop_r_max = crop_r_min + pixel_margin + roi_length - 1 + pixel_margin
        
        
                # Crop the original images
                # image_file = os.path.join(subject_dir, '{}_4d.nii.gz'.format(subject))
                subject_data = pydicom.read_file(os.path.join(subject_dir, os.listdir(subject_dir)[0]), force=True)
                    # instants = subject_data.CardiacNumberOfImages
                    
                h = subject_data.Rows #img_size[0]
                w = subject_data.Columns #img_size[1]
                image_data = np.zeros((w,h,slices,instants))
                
                # print(original_2D_paths[s])
                
                dir_cont = sorted(os.listdir(subject_dir), key=key_sort_files)
                for sl in range(slices):
                    # for i in [ed_instant, es_instant+1]:
                    # tmp = dir_cont[sl*instants : (sl+1)*instants]
                    for i in range(instants):
                        # print('Numbers: {:3.0f} {:2.0f} {:2.0f}'.format(i+sl*instants, i, sl))
                        # item = int(re.findall('[0-9]{1,3}' , dir_cont[i+sl*instants] )[0])
                        # if item != i+sl*instants+1:
                        #     print(item, i+sl*instants)
                        image_file = os.path.join(subject_dir, dir_cont[i+sl*instants])
                        # image_file = os.path.join(subject_dir, os.listdir(subject_dir)[i*slices+sl])
                        image_load = pydicom.read_file(image_file, force=True) 
                        # print(image_file)
                        try:
                            image_data[:,:,sl,i] = image_load.pixel_array
                        except:
                            print(image_file)

                
                # exit()
                original_r_min = max(0, crop_r_min)
                original_r_max = min(image_data.shape[0]-1, crop_r_max)
                original_c_min = max(0, crop_c_min)
                original_c_max = min(image_data.shape[1]-1, crop_c_max)
                crop_image_data = np.zeros((roi_length + 2 * pixel_margin, roi_length + 2 * pixel_margin,
                                            image_data.shape[2], image_data.shape[3]))
                crop_image_data[(original_r_min - crop_r_min):(original_r_max - crop_r_min + 1), 
                                (original_c_min - crop_c_min):(original_c_max - crop_c_min + 1), 
                                :, 
                                :] = \
                    image_data[original_r_min:(original_r_max + 1), 
                            original_c_min:(original_c_max + 1), 
                            :, 
                            :]
                crop_image_data = crop_image_data[::-1, ::-1, :, :]
                crop_image_file = os.path.join(out_dir, f'{dataset_name}_crop_2D', 'crop_{}_4d.nii.gz'.format(subject))
                nib.save(nib.Nifti1Image(crop_image_data, np.eye(4)), crop_image_file)
                
                
                
                
                # Crop the original labels
                if subject in train_subjects:
                    print(f"Training subjects: {subject}.")
                    for i in [ed_instant+1, es_instant+1]:
                        label_file = os.path.join(out_dir, 'MAD_OUS_gt', '{}_frame{}_gt.nii.gz'.format(subject,str(i).zfill(2)))
                        label_load = nib.load(label_file)
                        label_data = label_load.get_data()
                        crop_label_data = np.zeros((roi_length + 2 * pixel_margin, 
                            roi_length + 2 * pixel_margin,
                            image_data.shape[2]))
                        crop_label_data[(original_r_min - crop_r_min):(original_r_max - crop_r_min + 1), 
                                (original_c_min - crop_c_min):(original_c_max - crop_c_min + 1), 
                                :] = \
                            label_data[original_r_min:(original_r_max + 1), 
                            original_c_min:(original_c_max + 1), 
                            :]
                        crop_label_data = crop_label_data[::-1, ::-1, :]
                        crop_label_file = os.path.join(out_dir, 'MAD_OUS_crop_2D',
                            'crop_{}_frame{}_gt.nii.gz'.format(subject,str(i).zfill(2)))
                        nib.save(nib.Nifti1Image(crop_label_data, np.eye(4)), crop_label_file)
                
        
            
                # Save cropped 2D images
                crop_image_data = nib.load(crop_image_file).get_data()
        
                max_pixel_value = crop_image_data.max()
        
                if max_pixel_value > 0:
                    multiplier = 255.0 / max_pixel_value
                else:
                    multiplier = 1.0
        
                print('max_pixel_value = {}, multiplier = {}'.format(max_pixel_value, multiplier) )
                
                img_names = sorted(os.listdir(subject_dir), key=key_sort_files)
                all_files = [os.path.join(subject_dir, file) for file in img_names]
                for sl in range(slices):
                    # for i in [ed_instant, es_instant+1]:
                    for i in range(instants):
                        img_path = all_files[i+sl*instants]
                        img_path = img_path.replace('\\', '/')
                        s_t_image_file = re.sub(sub_key, sub_rep, img_path)
                        Image.fromarray((np.rot90(crop_image_data[:, ::-1, sl, i], 3) * multiplier).astype('uint8')).save(s_t_image_file + '.png')
                        
                        if i in [ed_instant, es_instant] and subject in train_subjects:
                            crop_label_file = os.path.join(out_dir, 'MAD_OUS_crop_2D', 
                                'crop_{}_frame{}_gt.nii.gz'.format(subject,str(i+1).zfill(2)))
                            crop_label_data = nib.load(crop_label_file).get_data()
                            s_t_label_file = s_t_image_file.replace('crop_', 'crop_gt_').replace('_gt_2D/', '_2D/')
                            # s_t_label_file = s_t_label_file
                            Image.fromarray((np.rot90(change_array_values(crop_label_data[:, ::-1, sl]), 3) * 50).astype('uint8')).save(s_t_label_file + '.png')
                # Save cropped 2D labels
                # if subject in train_subjects:
                #     for s in range(slices):
                #         # print(f"Training subjects: {subject}.")
                #         for t in [ed_instant, es_instant]:
                            
            
        
    

    print('Cropping done!')



if __name__ == '__main__':
    # crop_according_to_roi()
    # crop_according_to_roi('mesa')
    crop_according_to_roi('mad_ous')




