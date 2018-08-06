import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.decorators import image_comparison

@image_comparison(baseline_images=['Iris_Sepal_Length_BarPlot'],extensions=['png'])
def test_barplot_of_iris_sepal_length():

    # Write your functionality below
     fig = plt.figure(figsize=(8,6))
     ax = fig.add_subplot(111)
     species = ['setosa', 'versicolor', 'viriginica']
     index = [0.2, 1.2, 2.2]
     sepel_len = [5.01, 5.94, 6.59]
     ax.bar(index, sepel_len,
                color='red',width=0.5,edgecolor = 'black')
     ax.set_xlabel('Species')
     ax.set_ylabel('Sepal Length (cm)')
     ax.set_title('Mean Sepal Length of Iris Species')
     ax.set_xlim(0,3)
     ax.set_ylim(0,7)
     ax.set_xticks([0.45, 1.45,2.45],minor=False)
     ax.set_xticklabels(['setosa', 'versicolor', 'viriginica'])
     
@image_comparison(baseline_images=['Iris_Measurements_BarPlot'],extensions=['png'])
def test_barplot_of_iris_measurements():

    # Write your functionality below
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    sepal_len = [5.01, 5.94, 6.59]
    sepal_wd = [3.42, 2.77, 2.97]
    petal_len = [1.46, 4.26, 5.55]
    petal_wd = [0.24, 1.33, 2.03]
    species = ['setosa', 'versicolor', 'viriginica']
    species_index1 = [0.7, 1.7, 2.7]
    species_index2 = [0.9, 1.9, 2.9]
    species_index3 = [1.1, 2.1, 3.1]
    species_index4 = [1.3, 2.3, 3.3]
    
    ax.bar(species_index1,sepal_len,
                color='c',width=0.2,edgecolor = 'black',label='Sepal Length')
    ax.bar(species_index2, sepal_wd,
                color='m',width=0.2,edgecolor = 'black',label='Sepal Width')
     
    ax.bar(species_index3, petal_len,
                color='y',width=0.2,edgecolor = 'black',label='Petal Length')
     
    ax.bar(species_index4, petal_wd,
                color='orange',width=0.2,edgecolor = 'black',label='Petal Width')
    ax.set_xlabel('Species')
    ax.set_ylabel('Iris Measurements (cm)')
    ax.set_title('Mean Measurements of Iris Species')
    ax.set_xlim(0.5,3.7)
    ax.set_ylim(0,10)
    ax.set_xticks([1.1,2.1,3.1],minor=False)
    ax.set_xticklabels(['setosa', 'versicolor', 'viriginica'])
    ax.legend(loc='upper right')
    
'''    
 test_hbarplot_of_iris_petal_length

    Create a figure of size 12 inches in width, and 5 inches in height. Name it as `fig`.
    Create an axis, associated with figure `fig`,  using `add_subplot`. Name it as `ax`.
    Define a list `species`, with elements `['setosa', 'versicolor', 'viriginica']`. 
    Define a list `index`, with values `[0.2, 1.2, 2.2]`. 
    Define another list `petal_len` with values `[1.46, 4.26, 5.55]`. 
    These values represent Mean petal length of iris flowers belonging to three species.
    Draw a horizontal bar plot using `bar` function, such that width of each bar show mean 
    petal length of a species. 
    
    Use `index` and `petal_len` as variables. Set bar height to be `0.5`, color to be `c`,
    and border color of bar to be `black`.
    Label Y-Axis as `Species`
    Label X-Axis as `Petal Length (cm).
    Set Title as `Mean Petal Length of Iris Species`.
    Mark major ticks on Y-Axis at 0.45, 1.45, and 2.45.
    Label the major ticks on Y-Axis as `setosa`, `versicolor` and `viriginica` respectively.    
''' 
@image_comparison(baseline_images=['Iris_Petal_Length_BarPlot'],extensions=['png'])
def test_hbarplot_of_iris_petal_length():

    # Write your functionality below
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    species = ['setosa', 'versicolor', 'viriginica']
    index = [0.2, 1.2, 2.2]
    petal_len = [1.46, 4.26, 5.55]
    
    ax.barh(index, petal_len,
                color='c',height=0.5,edgecolor = 'black',label='Petal Width')
    ax.set_ylabel('Species')
    ax.set_xlabel('Petal Length (cm)')
    ax.set_title('Mean Petal Length of Iris Species')
    #ax.set_ylim(0.5,3.7)
    #ax.set_xlim(0,10)
    ax.set_yticks([0.45,1.45,2.45],minor=False)
    ax.set_yticklabels(species)