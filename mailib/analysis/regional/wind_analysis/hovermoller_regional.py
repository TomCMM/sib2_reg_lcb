#===============================================================================
# DESCRIPTION
#    Make the hovermoller plot with the wind and temeperature/humidity data from 
#    all the available stations data
#===============================================================================
from toolbox import PolarToCartesian
import pandas as pd
from stadata.lib.LCBnet_lib import att_sta
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from sklearn_pandas import DataFrameMapper
import seaborn as sns

def apply_func(df, func):
    index = df.index
    columns = df.columns
    df_trans = func(df.values)
    return  pd.DataFrame(df_trans, index=index, columns=columns)

if __name__ == '__main__':

    #===============================================================================
    # Read and preprocessing data
    #===============================================================================
    print "Reading data"
    # from the visual quality analysis
#     df_sm = pd.read_csv("/home/thomas/phd/framework/qa/out/qa_visual/Sm.csv", index_col=0, parse_dates=True)
#     df_dm = pd.read_csv("/home/thomas/phd/framework/qa/out/qa_visual/Dm.csv", index_col=0, parse_dates=True)
#     df_T = pd.read_csv("/home/thomas/phd/framework/qa/out/qa_visual/Ta.csv", index_col=0, parse_dates=True)
#     index = df_sm.index[(df_sm.index.isin(df_dm.index))&(df_sm.index.isin(df_T.index))] 
#     stanames =  df_sm.columns[(df_sm.columns.isin(df_dm.columns))&(df_sm.columns.isin(df_T.columns))]

#     df_sm = df_sm.loc[index,stanames]
#     df_dm = df_dm.loc[index,stanames]
#     df_T = df_T.loc[index,stanames]
# 
#     df_u = []
#     df_v = []
#     for norm,theta in zip(df_sm.values, df_dm.values):
#         U,V = PolarToCartesian(norm,theta)
#         df_u.append(U)
#         df_v.append(V)
#         
#     df_T = pd.DataFrame(df_T, index=index, columns =stanames).resample('H').mean()   
#     df_u = pd.DataFrame(df_u, index=index, columns =stanames).resample('H').mean()
#     df_v = pd.DataFrame(df_v, index=index, columns =stanames).resample('H').mean()
#         
    
    # from the filled data
    df_u = pd.read_csv("/home/thomas/phd/framework/fill/out/Uw.csv", index_col=0, parse_dates=True).resample('H').mean()   
    df_v = pd.read_csv("/home/thomas/phd/framework/fill/out/Vw.csv", index_col=0, parse_dates=True).resample('H').mean()   
    df_T = pd.read_csv("/home/thomas/phd/framework/fill/out/Ta.csv", index_col=0, parse_dates=True).resample('H').mean()   
    stanames = df_T.columns


    # group 
    df_T_daily=  df_T.groupby(lambda x: x.hour).mean()
    df_u_daily = df_u.groupby(lambda x: x.hour).mean()
    df_v_daily = df_v.groupby(lambda x: x.hour).mean()
    
    df_v_daily.dropna(inplace=True, axis=1)
    df_u_daily.dropna(inplace=True, axis=1)
    df_T_daily.dropna(inplace=True, axis=1)
    
    
#     df_T_daily = (df_T_daily - df_T_daily.mean()) / (df_T_daily.max() - df_T_daily.min())
#     df_v_daily = (df_v_daily - df_v_daily.mean()) / (df_v_daily.max() - df_v_daily.min())
#     df_u_daily = (df_u_daily - df_u_daily.mean()) / (df_u_daily.max() - df_u_daily.min())

    #======================================================================
    # Get distance from the Sea
    #======================================================================
    AttSta = att_sta("/home/thomas/phd/framework/predictors/out/att_sta/sta_att_distancetosea.csv")
#     AttSta = att_sta()
    dist_sea = AttSta.attributes.loc[:,'dist_sea'].copy()
    stanames = set(dist_sea.index) - (set(dist_sea.index) - set(stanames)) 
    dist_sea = dist_sea[stanames]
    dist_sea.sort()
    stanames = dist_sea.index
  
    #===============================================================================
    # Hovermoller plot
    #===============================================================================

    U = df_u_daily
    V = df_v_daily
    var = df_T_daily

    position, time = np.meshgrid(dist_sea.values, var.index)
   
    levels_contour=np.linspace(var.min().min(),var.max().max(),100)
    levels_colorbar=np.linspace(var.min().min(),var.max().max(),10)
   
    cmap = plt.cm.get_cmap("RdBu_r")
    fig = plt.figure()
 
    cnt = plt.contourf(position, time, var.loc[:,dist_sea.index].values,
                        levels = levels_contour, cmap=cmap)
    for c in cnt.collections:
        c.set_edgecolor("face")
         
    cbar = plt.colorbar(ticks=levels_colorbar)
    a = plt.quiver(position, time, U.loc[:,stanames].values, V.loc[:,stanames].values)
    cbar.ax.tick_params()
    qk = plt.quiverkey(a, 0.9, 1.05, 1, r'$1 \frac{m}{s}$',
                                    labelpos='E',
                                    fontproperties={'weight': 'bold'})
    cbar.set_label('Temperature C')
    plt.ylabel(r"Hours")
    plt.xlabel( r"Euclidian distance to the sea (Degree, 1degree ~ 100km)")
    plt.grid(True, color="0.5")
    plt.tick_params(axis='both', which='major')
#             plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    plt.title("Transect hovermoller ")
    plt.show()