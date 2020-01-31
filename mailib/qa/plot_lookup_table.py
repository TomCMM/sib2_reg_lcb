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
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 



def plot_heatmap(df, name, outpath=None):
#     yticks = df_outliers.index
#     keptticks = yticks[::int(len(yticks))]
#     yticks = ['' for y in yticks]
#     yticks[::int(len(yticks))] = keptticks
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
#     yticks = df_outliers.index
#     keptticks = yticks[::int(len(yticks))]
#     yticks = ['' for y in yticks]
#     yticks[::int(len(yticks))] = keptticks
    plt.figure(figsize=(11.69,8.27)) # for landscape
    g = sns.heatmap(df, linewidth=0.5, cmap='RdPu')
    
#     g.set_yticklabels(df_outliers.index,  fontsize = 6)
    plt.yticks(rotation=0, fontsize=6) 
    plt.xticks(rotation=90, fontsize=6)
    plt.title(var+ name, fontsize=12)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()



if __name__ == "__main__":
    from analysis.qa import outlier
    vars = ['Ta C', 'Ua %', 'Sm m/s', 'Dm G', 'Pa H','Qa g/kg','U m/s','V m/s']
    inpath = '../out/qa_visual/'
    outpath = "../res/"
    
    
    #    Serra Da Mantiquera
    Lat = [-25,-21]
    Lon = [-49, -43]
    params_selection={'Lat':Lat, 'Lon':Lon}
    
    
    for var in vars:
        print("Create quality control flag for the variable ->  " + str(var))
        f = glob.glob(inpath+ var[:2]+'*').pop()
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        
        df_available_data = df.groupby(lambda t: (t.year,t.month)).count().replace(0, np.nan)
        df_min = df.groupby(lambda t: (t.year,t.month)).min().replace(0, np.nan)#.subtract(df.groupby(lambda t: (t.year,t.month)).min().mean(axis=1), axis=0)
        df_mean = df.groupby(lambda t: (t.year,t.month)).mean().replace(0, np.nan)#.subtract(df.groupby(lambda t: (t.year,t.month)).mean().mean(axis=1), axis=0)
        df_max = df.groupby(lambda t: (t.year,t.month)).max().replace(0, np.nan)#.subtract(df.groupby(lambda t: (t.year,t.month)).max().mean(axis=1), axis=0)
        df_std = df.groupby(lambda t: (t.year,t.month)).aggregate(np.std).replace(0, np.nan)#.subtract(df.groupby(lambda t: (t.year,t.month)).aggregate(np.std).mean(axis=1), axis=0)
        df_outlier = df.groupby(lambda t: (t.year,t.month)).aggregate(outlier).replace(0, np.nan)
         
        with PdfPages(outpath + var[0:2] +'stats_per_month'+'.pdf') as pdf:
            print("Plotting ...")
            plot_heatmap_pdf(df_outlier, var, 'outlier per month',pdf)
            plot_heatmap_pdf(df_mean, var,'mean per month',pdf)
            plot_heatmap_pdf(df_std, var,'std per month',pdf)
            plot_heatmap_pdf(df_min, var,'min per month',pdf)
            plot_heatmap_pdf(df_max, var,'max per month',pdf)
            plot_heatmap_pdf(df_available_data, var,'available data',pdf)
