#===============================================================================
# DESCRIPTION
#    This script permit to select the stations and variables of interest and create a dataframe
#    this make the test of the model to be lighter and faster
#===============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib 
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import glob
import pandas as pd
import os
#from qa_check import outlier

matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8)



def plot_double_mass(df, ref_name=None, comp_name=None, outpath=False,var='var'):
    """
    Double mass plot

    :param df: Input dataframe
    :param ref_name: colmns name of the reference station
    :param comp_name:True, if True then apply on all the dataframe columns name
    :outpath Path of the folder to save the figure
    :return:
    """
    #
    # outpath = outpath + 'double_mass/' # Create a specific directory
    # assure_path_exists(outpath) # Check if the path exist

    # if comp_names:
    #     comp_names = df.columns.tolist()
    #     comp_names.remove(ref_name) # remove the reference from the list
    #
    # if not isinstance(comp_names, list):
    #     comp_names = [comp_names]
    #


    # for comp_name in comp_names:
    # try:
    print(f'Plotting {comp_name}')
    df_toplot = df.loc[:,[comp_name,ref_name]]
    df_toplot.columns = [comp_name, ref_name]

    df_toplot.dropna(how='any', inplace=True,axis=0)
    # df_toplot = df_toplot.resample('M').sum()

    df_toplot = df_toplot.cumsum()

    fig, ax1 = plt.subplots()
    plt.grid()
    ax1.plot(df_toplot.loc[:,comp_name], df_toplot.loc[:,ref_name])

    titlename =f'Plot {var} double mass of reference station {ref_name} compared to {ref_name} '
    plt.title(titlename)
    plt.xlabel(comp_name)
    plt.ylabel(ref_name)

    plt.plot(np.arange(df_toplot.loc[:,comp_name].min(), df_toplot.loc[:,comp_name].max()),c= 'k')
    ax2 = ax1.twinx()
    ax2.plot(df_toplot.loc[:,comp_name], df_toplot.loc[:,comp_name].index.year,alpha=0)
    ax2.set_ylabel('year', color='k')

    fig.tight_layout()
    # except ValueError:
    #     print('!!!!!!!!!!!!!!!!!!!')
    #     print(f'Could not plot {comp_name}')
    #     print('!!!!!!!!!!!!!!!!!!!')
    if not outpath:
        plt.show()
    else:
        print(f'print figure at {outpath}')
        plt.savefig(outpath+f'/double_mass_ref_{ref_name}_comp_{comp_name}.png')
    plt.close()

def plot_heatmap(df, name, outpath=None):
    plt.figure(figsize=(11.69,8.27)) # for landscape
    try:
        g = sns.heatmap(df, linewidth=0.5, cmap='RdPu')
        g.set_yticklabels(df.index,  fontsize = 3)
    except ValueError:
        print("Could not plot")
    
    plt.yticks(rotation=0, fontsize=3) 
    plt.xticks(rotation=90, fontsize=3)
    plt.title(name + "nb of stations= "+str(len(df.columns)) , fontsize=12)
    if outpath:
        plt.savefig(outpath+name+'.eps')  # saves the current figure into a pdf page
    else:
        plt.show()
    plt.close()


def plot_heatmap_pdf(df,var, name,pdf):
    plt.figure(figsize=(11.69,8.27)) # for landscape
    g = sns.heatmap(df, linewidth=0.5, cmap='RdPu')
    
#     g.set_yticklabels(df_outliers.index,  fontsize = 6)
    plt.yticks(rotation=0, fontsize=6) 
    plt.xticks(rotation=90, fontsize=6)
    plt.title(var+ name, fontsize=12)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()



if __name__ == "__main__":

    path_project = '/home/thomas/pro/research/stations_database/'
    inpath = path_project + '/2_dataframes/out/hourly/df_'
    outpath = path_project + "3_qa/res/hourly/"

    
    #    Serra Da Mantiquera
    Lat = [-25,-21]
    Lon = [-49, -43]


    vars = ['Ta C']

    params_selection={'lat':Lat, 'lon':Lon}

    kwargs_plots ={
        'ref_name':'A530',
    }

    main(vars,inpath, outpath, params_selection, kwargs_plots)
