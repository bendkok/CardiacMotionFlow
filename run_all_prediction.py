# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:09:48 2022

@author: benda
"""

from ROI.predict_roi_net import predict_roi_net
from ROI.crop_according_to_roi import crop_according_to_roi
from flow.predict_apparentflow_net import predict_apparentflow_net
from segmentation.predict_lvrv_net import predict_lvrv_net
from segmentation.calculate_dice_segmentation import calculate_dice_segmentation


def run_all_prediction(dataset='acdc'):
    
    print("Running predict_roi_net:")
    predict_roi_net(dataset=dataset)
    print("\n\n\nRunning crop_according_to_roi:")
    crop_according_to_roi(dataset=dataset)
    print("\n\n\nRunning predict_apparentflow_net:")
    predict_apparentflow_net(dataset=dataset)
    print("\n\n\nRunning predict_lvrv_net:")
    predict_lvrv_net(dataset=dataset)
    print("\n\n\nRunning calculate_dice_segmentation:")
    calculate_dice_segmentation(dataset=dataset)
    
    
    

if __name__ == '__main__':
    run_all_prediction()
    # run_all_prediction('mad_ous')