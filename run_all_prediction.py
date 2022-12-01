# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:09:48 2022

@author: benda
"""

from processing.preprocess_mad_ous import preprocess_mad_ous
from ROI.predict_roi_net import predict_roi_net
from ROI.crop_according_to_roi import crop_according_to_roi
from flow.predict_apparentflow_net import predict_apparentflow_net
from segmentation.predict_lvrv_net import predict_lvrv_net
from segmentation.calculate_dice_segmentation import calculate_dice_segmentation
from segmentation.make_overlay_segmentation import make_overlay_segmentation
from time import time


def run_all_prediction(dataset='acdc',
                       do_pre   = True,
                       do_roi   = True,
                       do_crop  = True,
                       do_flow  = True,
                       do_seg   = True,
                       do_dice  = True,
                       do_comp  = True,
                       ):
    
    start_time = time()
    # print(start_time)
    if dataset in ['mad_ous','mesa'] and do_pre:
        print("Running preprocess_mad_ous:")
        preprocess_mad_ous(dataset=dataset)
        print('\n\n')
    if do_roi:
        print("Running predict_roi_net:")
        predict_roi_net(dataset=dataset)
        print('\n\n')
    if do_crop:
        print("Running crop_according_to_roi:")
        crop_according_to_roi(dataset=dataset)
        print('\n\n')
    if do_flow:
        print("Running predict_apparentflow_net:")
        predict_apparentflow_net(dataset=dataset)
        print('\n\n')
    if do_seg:
        print("Running predict_lvrv_net:")
        predict_lvrv_net(dataset=dataset)
        print('\n\n')
    if do_dice:
        print("Running calculate_dice_segmentation:")
        calculate_dice_segmentation(dataset=dataset)
        print('\n\n')
    if do_comp:
        print("Running make_overlay_segmentation:")
        make_overlay_segmentation(dataset=dataset)
    
    end_time = time()
    
    print(f'\n\nTotal runtime: {end_time-start_time} sec.\n')
    
    

if __name__ == '__main__':
    # run_all_prediction()
    # run_all_prediction('mad_ous', do_flow=True)
    run_all_prediction('mesa', do_dice=True, do_flow=False)
    