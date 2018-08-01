'''
Define a numpy array 't' with expression 'np.arange(0.01, 5.0, 0.01)'. 
Define another numpy array 's1' with expression 'np.sin(2*np.pi*t)'
Define one more numpy array 's2' with expression 'np.sin(4*np.pi*t)'.
Create a figure of size 8 inches in width, and 6 inches in height. Name it as `fig`.
Create an axes, using `plt.subplot` function. Name it as `axes1`. 
The subplot must point to first virtual grid created by
 2 rows and 1 column. Set 'title' argument to 'Sin(2*pi*x)'.
Draw a line plot of 't' and 's1' using 'plot' function on 'axes1`.
Create another axes, using `plt.subplot` function. Name it as `axes2`. T
The subplot must point to second virtual grid created by 2 rows and 1 column. 
Set 'title' argument to 'Sin(4*pi*x)'. 
Set 'sharex' argument to 'axes1' and 'sharey' argument to 'axes1'.
Draw a line plot of 't' and 's2' using 'plot' function on 'axes2`.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.testing.decorators import image_comparison

#@image_comparison(baseline_images=['Multiple_Plots_Figure1'],extensions=['png'])
def test_generate_figure1():

    # Write your functionality below
    t = np.arange(0.01, 5.0, 0.01)
    s1 = np.sin(2*np.pi*t)
    s2 = np.sin(4*np.pi*t)
    fig = plt.figure(figsize=(8,6))
    axes1 = plt.subplot(2, 1, 1, title='Sin(2*pi*x)')
    
    axes2 = plt.subplot(2, 1, 2, title='Sin(4*pi*x)',sharex=axes1,sharey=axes1)
    axes1.plot(t, s1)
    axes2.plot(t, s2)
    plt.show()
test_generate_figure1()
'''
Set random seed to 1000 using the expression 'np.random.seed(1000)'.
Define a numpy array 'x' with expression 'np.random.rand(10)'.
Define another numpy array 'y' with expression 'np.random.rand(10)'.
Define one more numpy array 'z' with expression 'np.sqrt(x**2 + y**2)'.
Create a figure of size 8 inches in width, and 6 inches in height. Name it as `fig`.
Create an axes, using `plt.subplot` function. Name it as `axes1`. 
The subplot must point to first virtual grid created by 2 rows and 2 columns. 
Set 'title' argument to 'Scatter plot with Upper Traingle Markers'.

Draw a scatter plot of 'x' and 'y' using 'scatter' function on 'axes1`. Set argument 's' to 80, 'c' to 'z' and 'marker' to '^'.
Add ticks on X-Axis at 0.0, 0.4, 0.8, 1.2 and ticks on Y-Axis at -0.2, 0.2, 0.6, 1.0 respectively

Create an axes, using `plt.subplot` function. Name it as `axes2`. 
The subplot must point to Second virtual grid created by 2 rows and 2 columns. 
Set 'title' argument to 'Scatter plot with Plus Markers'.
Draw a scatter plot of 'x' and 'y' using 'scatter' function on 'axes2`. Set argument 's' to 80, 'c' to 'z' and 'marker' to '+'.
Add ticks on X-Axis at 0.0, 0.4, 0.8, 1.2 and ticks on Y-Axis at -0.2, 0.2, 0.6, 1.0 respectively

Create an axes, using `plt.subplot` function. Name it as `axes3`. 
The subplot must point to Third virtual grid created by 2 rows and 2 columns. 
Set 'title' argument to 'Scatter plot with Circle Markers'.
Draw a scatter plot of 'x' and 'y' using 'scatter' function on 'axes3`. 
Set argument 's' to 80, 'c' to 'z' and 'marker' to 'o'.
Add ticks on X-Axis at 0.0, 0.4, 0.8, 1.2 and ticks on Y-Axis at -0.2, 0.2, 0.6, 1.0 respectively

Create an axes, using `plt.subplot` function. Name it as `axes4`. 
The subplot must point to Fourth virtual grid created by 2 rows and 2 columns.
 Set 'title' argument to 'Scatter plot with Diamond Markers'.
Draw a scatter plot of 'x' and 'y' using 'scatter' function on 'axes4`. Set argument 's' to 80, 'c' to 'z' and 'marker' to 'd'.
Add ticks on X-Axis at 0.0, 0.4, 0.8, 1.2 and ticks on Y-Axis at -0.2, 0.2, 0.6, 1.0 respectively
Adjust the entire layout with expression 'plt.tight_layout()'.
'''
def test_generate_figure2():
    np.random.seed(1000)
    x = np.random.rand(10)
    y = np.random.rand(10)
    z = np.sqrt(x**2 + y**2)
    fig = plt.figure(figsize=(8,6))
    axes1 = plt.subplot(2, 2, 1, title='Scatter plot with Upper Traingle Markers')
    axes1.set_xticklabels([0.0, 0.4, 0.8, 1.2])
    axes1.set_yticklabels([-0.2, 0.2, 0.6, 1.0])
    axes1.scatter(x, y,s=80,c='z',marker = '^')
    

    axes2 = plt.subplot(2, 2, 2, title='Scatter plot with Plus Markers')
    axes2.scatter(x, y,s=80,c='z',marker = '+')
    axes2.set_xticklabels([0.0, 0.4, 0.8, 1.2])
    axes2.set_yticklabels([-0.2, 0.2, 0.6, 1.0])
    
    axes3 = plt.subplot(2, 2, 3, title='Scatter plot with Circle Markers')
    axes3.scatter(x, y,s=80,c='z',marker = 'o')
    axes3.set_xticklabels([0.0, 0.4, 0.8, 1.2])
    axes3.set_yticklabels([-0.2, 0.2, 0.6, 1.0])
    
    axes4 = plt.subplot(2, 2, 3, title='Scatter plot with Diamond Markers')
    axes4.scatter(x, y,s=80,c='z',marker = 'd')
    axes4.set_xticklabels([0.0, 0.4, 0.8, 1.2])
    axes4.set_yticklabels([-0.2, 0.2, 0.6, 1.0])

    plt.tight_layout()
    
test_generate_figure2()
@image_comparison(baseline_images=['Multiple_Plots_Figure2'],extensions=['png'])
def test_generate_figure2():

    # Write your functionality below


'''
Define a numpy array 'x' with expression 'np.arange(1, 101)'.
Define another numpy array 'y1' with expression 'y1 = x'.
Define another numpy array 'y2' with expression 'y1 = x**2'.
Define another numpy array 'y3' with expression 'y1 = x**3'.
Create a figure of size 8 inches in width, and 6 inches in height. Name it as `fig`.
Define a grid 'g' of 2 rows and 2 columns, using 'GridSpec' function. Make sure you have imported 'matplotlib.gridspec', before defining the grid.
Create an axes, using `plt.subplot` function. Name it as `axes1`. The subplot must span 1st row and 1st column of defined grid 'g'. Set 'title' argument to 'y = x'.
Draw a line plot of 'x' and 'y1' using 'plot' function on 'axes1`.
Create an axes, using `plt.subplot` function. Name it as `axes2`. The subplot must span 2nd row and 1st column of defined grid 'g'. Set 'title' argument to 'y = x**2'.
Draw a line plot of 'x' and 'y2' using 'plot' function on 'axes2`.
Create an axes, using `plt.subplot` function. Name it as `axes3`. The subplot must span all rows of 2nd column of defined grid 'g'. Set 'title' argument to 'y = x**3'.
Draw a line plot of 'x' and 'y3' using 'plot' function on 'axes3`.
Adjust the entire layout with expression 'plt.tight_layout()'.
'''

@image_comparison(baseline_images=['Multiple_Plots_Figure3'],extensions=['png'])
def test_generate_figure3():

    # Write your functionality below
    
test_generate_figure3()



fig = plt.figure(figsize=(10,8))
axes1 = plt.subplot(2, 1, 1, title='Plot1')
axes2 = plt.subplot(2, 1, 2, title='Plot2')
plt.show()
axes2 = plt.subplot(2, 2, 3, title='Plot2')
axes2.set_xticks([]); axes2.set_yticks([])
axes3 = plt.subplot(2, 2, 4, title='Plot3')
axes3.set_xticks([]); axes3.set_yticks([])
    