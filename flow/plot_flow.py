# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:52:40 2022

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="dark")

data = np.load('../acdc_info/acdc_data/patient146/predict_2D/flow2_03_00.npy')

print(data.shape)

x = np.linspace(0, 128, 128)
y = np.linspace(0, 128, 128)
 
X, Y = np.meshgrid(x, y)

plt.quiver(X,Y, data[:,:,0], data[:,:,1], color='darkred')
plt.show()



