# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:52:40 2022

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="dark")


def doplot(m):

    data = np.load('../acdc_info/acdc_data/patient101/predict_2D/flow2_04_{:02}.npy'.format(m))
    
    # sha = data.shape
    # print(sha)
    n = 2
    print(m, n)
    sha = data[::n,::n,:].shape
    
    x = np.linspace(0, sha[0], sha[0])
    y = np.linspace(0, sha[1], sha[1])
     
    X, Y = np.meshgrid(x, y)
    
    plt.quiver(X,Y, data[::n,::n,0], data[::n,::n,1], color='darkred')
    plt.show()


for i in range(30):    
    doplot(i)
