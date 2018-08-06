# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 08:19:28 2018

@author: baradhwaj
"""


Task 2 : Writing Functions

Complete the following function definitions, defined in `project/Matsession/tests/test_plots.py`, based on instructions provided below.
'''
1. test_sine_wave_plot 

    Create a figure of size 12 inches in width, and 3 inches in height. Name it as `fig`.
    Create an axis, associated with figure `fig`,  using `add_subplot`. Name it as `ax`.
    Create a numpy array `t` having 200 values between 0.0 and 2.0 . Use 'linspace' method to generate 200 values.
    Create a numpy array `v`, such that `v = np.sin(2.5*np.pi*t)`.
    Pass `t` and `v` as variables to `plot` function and draw a `red` line passing through choosen 200 points. Label the line as `sin(t)`.
    Label X-Axis as `Time (seconds)`
    Label Y-Axis as `Voltage (mV)`
    Set Title as `Sine Wave`
    Limit data on X-Axis from 0 to 2.
    Limit data on Y-Axis from -1 to 1.
    Mark major ticks on X-Axis at 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, and 2.0
    Mark major ticks on Y-Axis at -1, 0, and 1.
    Add a grid, whose linestyle is '--'.
    Add a legend
'''
import numpy as np
t = np.array(np.linspace(0.0, 5.0, num=20)) 
endpoint=False)
'''
2. test_multi_curve_plot

    Create a figure of size 12 inches in width, and 3 inches in height. Name it as `fig`.
    Create an axis, associated with figure `fig`,  using `add_subplot`. Name it as `ax`.
    Create a numpy array `x` having 20 values between 0.0 and 5.0 . Use 'linspace' method to generate 20 values.
    Create three numpy arrays `y1`, `y2` and `y3` using the expressions : `y1 = x`, `y2 = x**2` and `y3 = x**3` respectively.
    Draw a red colored line passing through `x` and `y1`, using `plot` function. Mark the 20 data points on the line as `circles`.Label the line as `y = x`.
    Draw a green colored line passing through `x` and `y2`, using `plot` function. Mark the 20 data points on the line as `squares`. Label the line as `y = x**2`.
    Draw a blue colored line passing through `x` and `y3`, using `plot` function. Mark the 20 data points on the line as `upward pointed triangles`. Label the line as `y = x**3`.
    Label X-Axis as `X`
    Label Y-Axis as `f(X)`
    Set Title as `Linear, Quadratic, & Cubic Equations`.
    Add a Legend.
'''


np.linspace(2.0, 3.0, num=5, endpoint=False)
'''
3. test_scatter_plot

    Create a figure of size 12 inches in width, and 3 inches in height. Name it as `fig`.
    Create an axis, associated with figure `fig`,  using `add_subplot`. Name it as `ax`.
    Consider the list `s = [50, 60, 55, 50, 70, 65, 75, 65, 80, 90, 93, 95]`. It represent the number of cars sold by a Company 'X' in each month of 2017, starting from Jan, 2017.
    Create a list, `months`, having numbers from 1 to 12.
    Draw a scatter plot with variables `months` and `s` as it's arguments. Mark the data points in `red` color. Use `scatter` function for plotting.
    Limit data on X-Axis from 0 to 13.
    Limit data on Y-Axis from 20 to 100.
    Mark ticks on X-Axis at 1, 3, 5, 7, 9, and 11.
    Label the X-Axis ticks as `Jan, Mar, May, Jul, Sep, and Nov` respectively.
    Label X-Axis as `Months`
    Label Y-Axis as `No. of Cars Sold`
    Set Title as "Cars Sold by Company 'X' in 2017".
 '''   
 
 import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.decorators import image_comparison

@image_comparison(baseline_images=['Sine_Wave_Plot'],extensions=['png'])
def test_sine_wave_plot():

    # Write your functionality below
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(111)
    t = np.array(np.linspace(0.0, 2.0, num=200))
    v = np.sin(2.5*np.pi*t)
    ax.plot(t,v,label='sin(t)',color='red')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('Sine Wave')
    ax.set_xlim(0,2)
    ax.set_ylim(-1,1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8,2.0],minor=False)
    ax.set_yticks([-1,0,1],minor=False)
    ax.grid(linestyle='--')
    ax.legend()
@image_comparison(baseline_images=['Multi_Curve_Plot'],extensions=['png'])
def test_multi_curve_plot():

    # Write your functionality below

    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(111)
    x = np.array(np.linspace(0.0, 5.0, num=20))
    y1 =   np.array(x)
    y2  =  np.array(x**2)
    y3  =  np.array(x**3)
    ax.plot(x, y1,marker = 'o',c='red',label='y = x')
    ax.plot(x, y2,marker = 's',c='green',label='y = x**2')
    ax.plot(x, y3,marker = '^',c='blue',label='y = x**3')
    ax.set_xlabel('X')
    ax.set_ylabel('f(X)')
    ax.set_title('Linear, Quadratic, & Cubic Equations')
    ax.legend(loc='upper right')

@image_comparison(baseline_images=['Scatter_Plot'],extensions=['png'])
def test_scatter_plot():
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(111)
    s = [50, 60, 55, 50, 70, 65, 75, 65, 80, 90, 93, 95]
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    plt.scatter(months ,s,color='red', edgecolors='black')
    ax.set_ylim(20,100)
    ax.set_xlim(0,13)
    ax.set_xticks([1,3,5,7,9,11],minor=False)
    ax.set_xticklabels(['Jan','Mar','May','Jul', 'Sep','Nov'])
    
    ax.set_xlabel('Months')
    ax.set_ylabel('No. of Cars Sold')
    ax.set_title("Cars Sold by Company 'X' in 2017")
    
    
    
    
    
    
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(111)
    x = np.array(np.linspace(0.0, 5.0, num=20))
    y1 =   np.array(x)
    y2  =  np.array(x**2)
    y3  =  np.array(x**3)
    ax.plot(x, y1,marker = 'o',c='red',label='y=x')
    ax.plot(x, y2,marker = 's',c='green',label='y=x**2')
    ax.plot(x, y3,marker = '^',c='blue',label='y=x**3')
    ax.set_xlabel('X')
    ax.set_ylabel('f(X)')
    ax.set_title('Linear, Quadratic, & Cubic Equations')
    ax.legend(loc='upper right')