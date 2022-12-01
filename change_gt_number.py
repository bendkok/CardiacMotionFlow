# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:49:50 2022

@author: benda
"""

import numpy as np
import nibabel as nib
import os

#loads the files
# folder = 'C:/Users/benda/Documents/Jobb_Simula/MAD_motion/MAD_OUS/MAD_OUS_gt'
folder = 'C:/Users/benda/Documents/Jobb_Simula/MAD_motion/MESA/MESA_gt'
fil = os.listdir(folder)
files = [os.path.join(folder, file) for file in fil]


for file in files:
    
    #loads data for current file
    nib_object = nib.load(file)
    data = nib_object.get_fdata()
    
    data[np.where(data > 0)] += 3
    data[np.where(data == 6)] = 1
    data[np.where(data == 4)] = 2
    data[np.where(data == 5)] = 3
    
    nib.save(nib.Nifti1Image(data, np.eye(4)), file)
    # nib.save(data, file)

print(files)


# nib.Nifti1Image()
