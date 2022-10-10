""" A function to generate the lists of files for ROI-net inference"""

import sys
sys.path.append('..')

import os
import pydicom


def data_mesa_roi_predict():
    data_dir = "C:\\Users\\benda\\Documents\\Jobb_Simula\\MAD_motion\\MESA_set1_sorted\\{}" #config.acdc_data_dir

    
    predict_img_list = []
    predict_gt_list = []
    subject_dir_list = []
    
    skipped = 0
    # all_subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(1, 151)]
    all_subjects = ['MES00{}01'.format(str(x).zfill(3)) for x in range(100)]
    for subject in all_subjects:
        subject_dir = data_dir.format(subject)
        # print(subject_dir, subject)
        subject_dir_list.append(subject_dir)
            


        # original_2D_path = os.path.join(subject_dir, 'original_2D')
        #there were at least five different naming conventions and some patient don't have a short-axis scan
        # original_2D_path = os.path.join(subject_dir, 'short_axis_cine_(multi-slice)')
        if os.path.exists(os.path.join(subject_dir, 'short_axis_cine_(multi-slice)') ):
            original_2D_path = os.path.join(subject_dir, 'short_axis_cine_(multi-slice)')
        elif os.path.exists(os.path.join(subject_dir, 'short_axis_(10_slices)') ):
            original_2D_path = os.path.join(subject_dir, 'short_axis_(10_slices)')
        elif os.path.exists(os.path.join(subject_dir, 'short_axis_cine_(multi-slice') ):
            original_2D_path = os.path.join(subject_dir, 'short_axis_cine_(multi-slice')
        elif os.path.exists(os.path.join(subject_dir, 'short_axis_tag_90') ):
            original_2D_path = os.path.join(subject_dir, 'short_axis_tag_90')
        elif os.path.exists(os.path.join(subject_dir, 'derived-primary-other\\short_axis_(10_slices)') ):
            original_2D_path = os.path.join(subject_dir, 'derived-primary-other\\short_axis_(10_slices)')
        else: #if not os.path.exists(original_2D_path):
            print("Warning! No short-axis scan found for subject {}.".format(subject))
            skipped += 1
            continue
            # return [],[]

        #some patients have mulitple series, finds the one with the most slices
        if os.path.exists(os.path.join(original_2D_path, 'series_0')):
            subdirs = os.listdir(original_2D_path)
            # dir_lens = []
            longest = len(os.listdir(os.path.join(original_2D_path, subdirs[0])))
            longest_dir = subdirs[0]
            for s in subdirs[1:]:
                curr = len(os.listdir(os.path.join(original_2D_path, s)))
                if curr > longest:
                    longest = curr
                    longest_dir = s
                # dir_lens.append(len(os.listdir(s)))
            original_2D_path = os.path.join(original_2D_path, longest_dir)
            # print(original_2D_path, subdirs)
        
        # else:
            
        #where we'll store the masks
        subject_predict_dir = os.path.join(subject_dir, 'mask_original_2D')
        if not os.path.exists(subject_predict_dir):
            os.makedirs(subject_predict_dir)
            
            
        #a list with all the frames for this patient
        all_frames = os.listdir(original_2D_path)
        
        #it seems like each patient has 20 frames for each slice, hope that's consistent
        #this way we don't need to use re or similar to get the desired files
        subject_data = pydicom.read_file(os.path.join(original_2D_path, all_frames[0]))
        instants = subject_data.CardiacNumberOfImages
        slices = int(len(all_frames)/instants)
        if instants != 20:    
            print(subject, instants, slices)

        # Prediction on the ED stacks only
        used_instants = [0] #[ed_instant]
        #Here they tried to get only the data for ED. MESA dosen't record when that is, so I've just used 0 instead
        
        # for s in range(slices):
        for s in range(int(round(slices * 0.1 + 0.001)), int(round(slices * 0.5 + 0.001))): #they use this long thing to decide which slices to use, not sure why
            for t in used_instants: #there's only one element here, but that might change in the future
                s_t_image_file = os.path.join(original_2D_path, all_frames[s*instants+t]) #select the desired frame
                predict_img_list.append(s_t_image_file)
        

    predict_gt_list = ['']*len(predict_img_list) #this needs to be included
    print('\npredict_image_count = {}. '.format(len(predict_img_list)) )
    print(f"Skipped patients = {skipped}.\n")

    return predict_img_list, predict_gt_list, subject_dir_list



if __name__ == '__main__':
    out = data_mesa_roi_predict()
    # print(out)
    # for o in out[0]:
        # print(o)


