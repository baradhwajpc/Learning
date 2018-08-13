# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 21:55:58 2018

@author: baradhwaj
"""
'''
1. test_hist_of_a_sample_normal_distribution 

    Create a figure of size 8 inches in width, and 6 inches in height. Name it as `fig`.
    Create an axis, associated with figure `fig`,  using `add_subplot`. Name it as `ax`.
    Set random seed to 100 using the expression `np.random.seed(100)`.
    Create a normal distribution `x1` of 1000 values, with mean 25 and standard deviation 3.0 . 
    Use `np.random.randn`.
    Draw a histogran of `x1` with 30 bins. Use `hist` function.
    Label X-Axis as `X1`
    Label Y-Axis as `Bin Count`
    Set Title as `Histogram of a Single Dataset`.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.decorators import image_comparison

@image_comparison(baseline_images=['Histogram'],extensions=['png'])
def test_hist_of_a_sample_normal_distribution():
    # Write your functionality belowhttps://www.hackerrank.com/tests/#
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    np.random.seed(100)
    #x1 = np.random.randn(25,3,1000)
    mu, sigma = 25, 3.0
    x1 = mu + sigma*np.random.randn(1000)
    plt.hist(x1,bins=30)
    plt.xlabel('X1')
    plt.ylabel('Bin Count')
    plt.title('Histogram of a Single Dataset')    
'''
2. test_boxplot_of_four_normal_distribution

    Create a figure of size 8 inches in width, and 6 inches in height. Name it as `fig`.
    Create an axis, associated with figure `fig`,  using `add_subplot`. Name it as `ax`.
    Set random seed to 100 using the expression `np.random.seed(100)`.
    Create a normal distribution `x1` of 1000 values, with mean 25 and standard deviation 3.0 . 
    Use `np.random.randn`.
    Create a normal distribution `x2` of 1000 values, with mean 35 and standard deviation 5.0 . 
    Use `np.random.randn`.
    Create a normal distribution `x3` of 1000 values, with mean 55 and standard deviation 10.0 .
    Use `np.random.randn`.
    Create a normal distribution `x4` of 1000 values, with mean 45 and standard deviation 3.0 . 
    Use `np.random.randn`.
    Create a list `labels` with elements `['X1', 'X2', 'X3', 'X4]`.
    Draw a Boxplot of `x1, x2, x3, x4` with notches and label it using `labels` list. 
    Use `boxplot` function. 
    Choose `+` symbol for outlier and fill color inside boxes by setting `patch_artist` 
    argument to `True`.
    Label X-Axis as `Dataset`
    Label Y-Axis as `Value`
    Set Title as `Box plot of Multiple Datasets`.
'''

@image_comparison(baseline_images=['Boxplot'],extensions=['png'])
def test_boxplot_of_four_normal_distribution():
    # Write your functionality below
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    np.random.seed(100)
    mu, sigma = 25, 3.0
    x1 = mu + sigma*np.random.randn(1000)
    mu, sigma = 35, 5.0
    x2 = mu + sigma*np.random.randn(1000)
    mu, sigma = 55, 10.0
    x3 = mu + sigma*np.random.randn(1000)
    mu, sigma = 45, 3.0
    x4 = mu + sigma*np.random.randn(1000)
    labels = ['X1', 'X2', 'X3', 'X4']
    toPlot = [x1,x2,x3,x4]
    ax.boxplot(toPlot,patch_artist=True, sym='+',notch=True,labels=labels)
    plt.xlabel('Dataset')
    plt.ylabel('Value')
    plt.title('Box plot of Multiple Datasets')
