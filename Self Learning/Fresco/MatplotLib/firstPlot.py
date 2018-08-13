# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 09:11:06 2018

@author: baradhwaj
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

@image_comparison(baseline_images=['My_First_plot'],extensions=['png'])
def test_my_first_plot():
    
    # Write your functionality below
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    t = [5, 10, 15, 20, 25]
    d =[25, 50, 75, 100, 125]
    plt.plot(t,d,label='d = 5t')
    plt.xlabel('time (seconds)')
    plt.ylabel('distance (meters)')
    plt.title('Time vs Distance Covered')
    ax.set_xlim(0,30)
    ax.set_ylim(0,130)
    ax.legend()
