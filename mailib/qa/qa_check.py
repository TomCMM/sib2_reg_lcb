#===============================================================================
# DESCRIPTION
#     This module contains function for quality check assurance inspired from 
#    Durre and al.
#    If ran it filter the data with the qa function, plot a resume of the percetnage of removed value.
#    It also save the time serie of removed value in the qs/res folder
#    Finally it save the filtered data in the qa/out folder.
#===============================================================================
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.qa import plot_heatmap
from stadata.lib.LCBnet_lib import  att_sta
from toolbox.geo import PolarToCartesian
from toolbox.meteo import q
from toolbox.tools import common_index


def outlier(df):
    df = (df-df.mean())/df.std()
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_s = ((df < (Q1 - 2 * IQR)) | (df > (Q3 + 2 * IQR)))
    return outlier_s.sum()

def qa_flag_outlier(df):
    df = (df-df.mean())/df.std()
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_s = ((df < (Q1 - 2 * IQR)) | (df > (Q3 + 2 * IQR)))
    return outlier_s

def qa_flag_spatial_outlier(df):
    """
    
    This function is very slow
    """
    df = (df-df.mean())/df.std()
    df = df.T
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_s = ((df < (Q1 - 2 * IQR)) | (df > (Q3 + 2 * IQR)))
    return outlier_s.T

def qa_flag_360(df):
    """
    return a mask which identify zero
    """
    return df == 360

def qa_flag_0(df):
    """
    return a mask which identify zero
    """
    return df == 0

def qa_flag_100(df):
    """
    return a mask which identify zero
    """
    return df == 100

def qa_flag_repetition(df):
    """
    Return a flag for repetition
    """
    diff1 = df.diff(1)
    diff2 = df.diff(2)
    diff3 = df.diff(3)
    diff = diff1 + diff2 + diff3
    
    return diff == 0 

def limit_n_month(df,n_month):
    """
    return mask with the station which have at leat n_month of data
    """
    n_events = n_month*30*24
    mask_stations =  df.count() > n_events
    return mask_stations

def limit_nbevent_day(df, n_hours=4):
    """
    Return:
        Mask with True if their at least n_events per day
    """
    nb_events_day = df.resample('D').count()
    nb_events_day = nb_events_day.reindex(index = df.index, method='pad')
    nb_events_day = nb_events_day.replace(0,np.nan)
    return nb_events_day < n_hours 

def apply_mask(df, mask_functions, plot_removed=False, outpath_plot_remove = 'plot_timeserie_flagged_values/'):
    """
    PARAMETERS:
        df: a dataframe object with the station variables
        mask_function: a dictionnary of flag function to be applied
    RETURN
        df, filtred data frame 
        masks, a dictionary of mask array
        df_qa_count: The percentage of filtred data for each stations
    
    """
    print "000"*10
    print "APPLYING FILTER"
    print "000"*10
    
    
    df_qa_count = {}
    masks = {}
    
    
    for mask_name in mask_functions.keys():
        print "Apply -> " + mask_name
        mask_function = mask_functions[mask_name]
        df_mask= mask_function(df) # apply the mask function
        masks[mask_name] = df_mask # keep the masked dataframe in a dictionary
        df_qa_count[mask_name] = df_mask[df_mask==True].count() / df.count()*100 # count the frequency of filtered data
    
    df_qa_count= pd.DataFrame(df_qa_count)
    if plot_removed == True:
        for mask_name in masks.keys(): # apply mask and filter the data
            print "plot -->" + mask_name
            df[masks[mask_name]].plot()
            plt.savefig(output_summary_path +outpath_plot_remove +var[:2]+'_'+mask_name+".png") # plot removed values
                
    for mask_name in masks.keys(): # apply mask and filter the data
        df[masks[mask_name]] = np.nan # apply the mask
    
    return df, masks, df_qa_count


exceptions_mask_var = {'Pa H':['outlier'],
                       'Dm G':['outlier'],
                       'Sm m/s':['outlier']}


def apply_exception_mask(var ,exceptions_mask_var, mask_functions):
    """
    Remove a mask is their is an exception
    
    """
    print 
    if var in exceptions_mask_var.keys():
        for exmask in exceptions_mask_var[var]:
            try:
                del mask_functions[exmask]
            except:
                print "No " + str(exmask)    
    return mask_functions


    
def getdf_qa_visual(inpath, var):
    file= glob.glob(inpath+var[:2]+'*').pop()
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    df.dropna(inplace=True, axis=0, how='all')
    df.dropna(inplace=True, axis=1, how='all')
    stanames_visual_qa =  attsta_visual_qa.stations(values=[], params={var:[0,2]})
    df_select = df.loc[:,stanames_visual_qa]
    return df_select


def main_qa(df, var):
    """
    Description:
        Main of the Qualtity Analysis
    
    """
    #=======================================================================
    # Select stations with more than 3 months of data
    #=======================================================================
    df.dropna(inplace=True, axis=1, how='all') # drop when there is no stations data
    mask_month = limit_n_month(df, n_month=3)
    df_filter = df.loc[:,mask_month]
             
    #=======================================================================
    # Select mask_functions
    #=======================================================================
    mask_function_repetition = { 'events_per_day':limit_nbevent_day, '360':qa_flag_360, '100':qa_flag_100,
                                 '0':qa_flag_0,'repetition':qa_flag_repetition} # Filter function
    
#         mask_function_repetition = {'100':qa_flag_100, 'repetition':qa_flag_repetition} # Filter function
    mask_function_outlier = {'outlier':qa_flag_outlier}


    mask_functions_repetition = apply_exception_mask(var ,exceptions_mask_var, mask_function_repetition)
    mask_functions_outlier = apply_exception_mask(var ,exceptions_mask_var, mask_function_outlier)
    
    #=======================================================================
    # Apply mask
    #=======================================================================
    df_filtred_repetition, masks_repetition, df_qa_count_repetition = apply_mask(df_filter, mask_functions_repetition, plot_removed=False)
    df_filtred_outlier, masks_outlier, df_qa_count_outlier = apply_mask(df_filtred_repetition, mask_function_outlier, plot_removed=False)
#          
#         #===============================================================================
#         # Plot summaries
#         #===============================================================================
    plot_heatmap(df_qa_count_repetition, var[:2]+'Total_count_of_filtered_data_repetition' ,output_summary_path)
    plot_heatmap(df_qa_count_outlier, var[:2]+'Total_count_of_filtered_data_outlier' ,output_summary_path)   
    df_filtred_outlier.to_csv(outpath+'qa_visual_mask/'+var[:2]+".csv") # save the filtered data in the qs/out/ folder
    df_filter.to_csv(outpath+'qa_visual/'+var[:2]+".csv") # save the filtered data in the qs/out/ folder


if __name__ == '__main__':
    inpath = "../in/"
    output_summary_path = '../res/'
    outpath = '../out/'
    metadata_visual_qa = '/home/thomas/phd/obs/staClim/metadata/metadata_allnet_qa_visual_v3.csv'
    attsta_visual_qa = att_sta(metadata_visual_qa)
    
    vars = ['Dm G', 'Ua %', 'Sm m/s', 'Ta C', 'Pa H']
    for var in vars:
        print 'Applying mask for the variable '+ str(var)
        df = getdf_qa_visual(inpath,var)
        main_qa(df, var)

    df_ta =getdf_qa_visual(outpath+'qa_visual/','Ta C')
    df_ua =getdf_qa_visual(outpath+'qa_visual/','Ua %')
    df_pa =getdf_qa_visual(outpath+'qa_visual/','Pa H')
    df_dm = getdf_qa_visual(outpath+'qa_visual/','Dm G')
    df_sm = getdf_qa_visual(outpath+'qa_visual/','Sm m/s')

    index =pd.DatetimeIndex(common_index(df_dm.index, df_sm.index))
    stanames =  common_index(df_dm.columns, df_sm.columns)
# 
#     #===========================================================================
#     # Write specific humidity
#     #===========================================================================
    df_q = q(df_ua, df_ta, df_pa)
    main_qa(df_q, 'Qa g/kg')

    #===========================================================================
    # Write wind speed components
    #===========================================================================
    df_sm = df_sm.loc[index,stanames]
    df_dm = df_dm.loc[index,stanames]

    df_u =[]
    df_v = []
    for sta in stanames:
        wind = pd.concat([df_sm.loc[:,sta], df_dm.loc[:,sta]], join='inner', axis=1)
        U,V = PolarToCartesian(wind.iloc[:,0],wind.iloc[:,1])
        df_u.append(U)
        df_v.append(V)
        
    df_u = pd.concat(df_u, join='outer', axis=1)
    df_v = pd.concat(df_v, join='outer', axis=1)
    main_qa(df_u, 'Uw m/s')
    main_qa(df_v, 'Vw m/s')    
        
        
        
        
        
