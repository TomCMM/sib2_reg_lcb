# ===============================================================================
# DESCRIPTION
#     This module contains function for quality check assurance inspired from  Durre and al. 2014
#    If ran it filter the data with the qa function, plot a resume of the percetnage of removed value.
#    It also save the time serie of removed value in the qs/res folder
#    Finally it save the filtered data in the qa/out folder.
# ===============================================================================


import pandas as pd
import glob
import matplotlib.pyplot as plt
# from stanet.lcbstanet import Att_Sta
#from plot_df_stats_visu import plot_heatmap
import numpy as np
#from main_lib.toolbox.tools import check_directory_exist


def outlier(df):
    df = (df - df.mean()) / df.std()
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_s = ((df < (Q1 - 3 * IQR)) | (df > (Q3 + 3 * IQR)))
    return outlier_s.sum()


def qa_flag_outlier(df):
    df = (df - df.mean()) / df.std()
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_s = ((df < (Q1 - 3 * IQR)) | (df > (Q3 + 3 * IQR)))
    return outlier_s


def qa_flag_spatial_outlier(df):
    """

    This function is very slow
    """
    df = (df - df.mean()) / df.std()
    df = df.T
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_s = ((df < (Q1 - 3 * IQR)) | (df > (Q3 + 3 * IQR)))
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


def limit_n_month(df, n_month):
    """
    return mask with the station which have at leat n_month of data
    """
    n_events = n_month * 30 * 24
    mask_stations = df.count() > n_events
    return mask_stations


def limit_nbevent_day(df, n_hours=4):
    """
    Return:
        Mask with True if their at least n_events per day
    """
    nb_events_day = df.resample('D').count()
    nb_events_day = nb_events_day.reindex(index=df.index, method='pad')
    nb_events_day = nb_events_day.replace(0, np.nan)
    return nb_events_day < n_hours


def apply_mask(df, mask_functions, plot_removed=False, outpath_plot_remove='plot_timeserie_flagged_values/'):
    """
    PARAMETERS:
        df: a dataframe object with the station variables
        mask_function: a dictionnary of flag function to be applied

    RETURN
        df, filtred data frame
        masks, a dictionary of mask array
        df_qa_count: The percentage of filtred data for each stations

    """
    print("000" * 10)
    print("APPLYING FILTER")
    print("000" * 10)

    df_qa_count = {}
    masks = {}

    for mask_name in mask_functions.keys():
        print("Apply -> " + mask_name)
        mask_function = mask_functions[mask_name]
        df_mask = mask_function(df)  # apply the mask function
        masks[mask_name] = df_mask  # keep the masked dataframe in a dictionary
        df_qa_count[mask_name] = df_mask[
                                     df_mask == True].count() / df.count() * 100  # count the frequency of filtered data

    df_qa_count = pd.DataFrame(df_qa_count)
    if plot_removed == True:
        for mask_name in masks.keys():  # apply mask and filter the data
            print("plot -->" + mask_name)
            df[masks[mask_name]].plot()
            plt.savefig(output_figure_path + outpath_plot_remove + var[:2] + '_' + mask_name + ".png")  # plot removed values

    for mask_name in masks.keys():  # apply mask and filter the data
        df[masks[mask_name]] = np.nan  # apply the mask

    return df, masks, df_qa_count


def remove_exception_mask(var, exceptions_mask_var, mask_functions):
    """
    Remove a mask is their is an exception

    """
    print
    if var in exceptions_mask_var.keys():
        for exmask in exceptions_mask_var[var]:
            try:
                del mask_functions[exmask]
            except:
                print("No " + str(exmask))
    return mask_functions


def getdf_manual_total_exclusion(inpath, var, attsta_total_exclusion_byvar):
    # file= glob.glob(inpath+var[:2]+'*').pop()
    df = pd.read_csv(inpath, index_col=0, parse_dates=True)
    df.dropna(inplace=True, axis=0, how='all')
    df.dropna(inplace=True, axis=1, how='all')
    excluded_staid = attsta_total_exclusion_byvar.get_sta_id_in_metadata(values=[], params={var: [0, 2]})
    df_select = df.drop(excluded_staid, axis=1)
    return df_select


def Run_Qa(df_filtred, var):
    """
    Description:
        Apply the quality control and filtering procedure:
            1) Select stations with more than 3 months of data
            2) Select mask_functions and apply exception to the variables if necessary
            3) Apply masks
            4) Plot summaries
            5) Write manual and manual_and_auto filtering

    """
    # =======================================================================
    # 1) Select stations with more than 3 months of data
    # =======================================================================
    df_filtred.dropna(inplace=True, axis=1, how='all')  # drop when there is no stations data
    mask_month = limit_n_month(df_filtred, n_month=3)
    df_filter = df_filtred.loc[:, mask_month]

    # =======================================================================
    # 2) Select mask_functions and apply exception
    # =======================================================================
    mask_function = {'events_per_day': limit_nbevent_day, '360': qa_flag_360, '100': qa_flag_100,
                     '0': qa_flag_0, 'repetition': qa_flag_repetition,'outlier':qa_flag_outlier}  # Filter function

    exceptions_mask_var = {'Pa H': ['outlier'],
                           'Dm G': ['outlier'],
                           'Sm m/s': ['outlier'],
                           'U m/s': ['outlier'],
                           'V m/s': ['outlier'],
                           }
    # exceptions_mask_var={}
    mask_functions = remove_exception_mask(var, exceptions_mask_var, mask_function)

    # =======================================================================
    # 3)  Apply mask
    # =======================================================================
    df_filtred, masks, df_qa_count = apply_mask(df_filter, mask_functions, plot_removed=False)


    if var == 'Ta C':
        print('AHHH'*100)
        df_filtred.loc["2015-07-05":"2015-07-12",'rib_C14'] =np.nan
        df_filtred.loc["2015-04-10":"2015-04-12",'rib_C14'] =np.nan
        df_filtred.loc["2015-04-25":"2015-05-01",'rib_C10'] =np.nan
        df_filtred.loc["2015-03-09":"2015-03-13",'rib_C18'] =np.nan
        df_filtred.loc["2015-05-10":"2015-05-13",'rib_C18'] =np.nan
        df_filtred.loc["2015-09-10":"2015-09-13",'rib_C18'] =np.nan
        df_filtred.loc["2016-01-07":"2015-01-09",'rib_C18'] =np.nan
        df_filtred.loc["2015-04-10":"2015-04-12",'rib_C18'] =np.nan
        df_filtred.loc["2015-05-10":"2015-05-12",'rib_C11'] =np.nan
        df_filtred.loc["2015-06-10":"2015-06-12", 'rib_C18'] =np.nan
        df_filtred.loc["2015-07-11":"2015-07-12", 'rib_C18'] =np.nan
        df_filtred.loc["2015-07-19":"2015-09-21", 'rib_C10'] =np.nan
        df_filtred.loc["2015-08-10":"2015-08-11", 'rib_C18'] =np.nan
        df_filtred.loc["2015-08-23":"2015-08-24", 'rib_C10'] =np.nan
        df_filtred.loc["2015-10-21":"2015-11-01", 'rib_C18'] =np.nan
        df_filtred.loc["2015-11-10":"2015-11-14", 'rib_C18'] =np.nan
        df_filtred.loc["2015-12-10":"2015-12-14", 'rib_C18'] =np.nan
        df_filtred.loc["2015-09-13":"2015-09-14", 'rib_C15'] =np.nan
        df_filtred.loc["2016-02-18":"2016-02-19", 'rib_C08'] =np.nan
        df_filtred.loc["2016-03-02":"2016-03-04", 'rib_C18'] =np.nan


    if var == 'Qa g/kg':
        df_filtred.loc["2015-04-25":"2015-05-01", 'rib_C10'] =np.nan
        df_filtred.loc["2015-06-11":"2015-06-13", 'rib_C18'] =np.nan
        df_filtred.loc["2015-07-05":"2015-07-07", 'rib_C14'] =np.nan
        df_filtred.loc["2015-07-18":"2015-07-21", 'rib_C10'] =np.nan
        df_filtred.loc["2015-08-10":"2015-08-12", 'rib_C18'] =np.nan
        df_filtred.loc["2015-08-23":"2015-08-24", 'rib_C10'] =np.nan
        df_filtred.loc["2016-02-18":"2016-02-19", 'rib_C10'] =np.nan
        df_filtred.loc["2015-04-10":"2015-04-12", 'rib_C18'] =np.nan
        df_filtred.loc["2015-07-10":"2015-07-12", 'rib_C18'] =np.nan
        df_filtred.loc["2015-09-13":"2015-09-14", 'rib_C15'] =np.nan
        df_filtred.loc["2015-09-27":"2015-10-12", 'rib_C13'] =np.nan
        df_filtred.loc["2015-11-15":"2016-04-01", 'rib_C16'] =np.nan


    return df_filtred, mask_function, df_qa_count
