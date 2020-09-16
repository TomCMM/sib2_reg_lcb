#===============================================================================
# DESCRIPTION
#    Calculate the transport of temperature, specific humidity in the Ribeirao Das Posses
#===============================================================================

from clima_lib.LCBnet_lib import LCB_net, att_sta
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


def transport(df_T, df_Q, df_U, df_V):
    df_T_mean = df_T.mean()
    df_U_mean = df_U.mean()
    df_V_mean = df_V.mean()
    df_Q_mean = df_Q.mean()
    
    T_var = df_T - df_T_mean
    Q_var = df_Q - df_Q_mean
    U_var = df_U - df_U_mean
    V_var = df_V - df_V_mean
    
    covUT = (T_var *  U_var)
    covVT = (T_var *  V_var)
    covUQ = (Q_var *  U_var)
    covVQ = (Q_var *  V_var)
    
    
    UT = df_T_mean*df_U_mean + covUT.mean()
    VT = df_T_mean*df_V_mean + covVT.mean()
    
    UQ = df_Q_mean*df_U_mean + covUQ.mean()
    VQ = df_Q_mean*df_V_mean + covVQ.mean()
    
    df_cov = pd.concat([covUT, covVT, covUQ, covVQ], axis=1, join='inner')
    df_cov.columns = ['covUT', 'covVT', 'covUQ', 'covVQ']
    return df_cov
    

if __name__ =='__main__': 

#===============================================================================
# USER input 
#===============================================================================
    Path_LCB = '/home/thomas/phd/obs/lcbdata/obs/full_sortcorr/'
    path_att = '/home/thomas/phd/local_downscaling/data/metadata/df_topoindex_ribeirao.csv'

    From = "2015-03-01 00:00:00"
    To = "2016-01-01 00:00:00"

#===============================================================================
# GET DATA
#===============================================================================
    net_LCB = LCB_net()

    AttSta = att_sta()
    AttSta.attributes['Alt'] = AttSta.attributes['Alt'].astype(float) # if not give problem in the stepwise libnear regression 
    AttSta.attributes.dropna(how='all',axis=1, inplace=True)
    AttSta.attributes.dropna(how='any',axis=0, inplace=True)

    AttSta.setInPaths(Path_LCB)
    stanames_LCB = AttSta.get_sta_id_in_metadata(values = ['Ribeirao'])
    [stanames_LCB.remove(x) for x in ['C11','C12'] if x in stanames_LCB ] # Remove stations
    Files_LCB =AttSta.getatt(stanames_LCB,'InPath')
    net_LCB.AddFilesSta(Files_LCB)
    

    
    df_T_H = net_LCB.getvarallsta(var='Ta C', by='H', From=From, To = To ).resample('H').mean().mean(axis=1)
    df_Q_H = net_LCB.getvarallsta(var='Ua g/kg', by='H', From=From, To = To ).resample('H').mean().mean(axis=1)
    df_U_H = net_LCB.getvarallsta(var='U m/s', by='H', From=From, To = To ).resample('H').mean().mean(axis=1)
    df_V_H = net_LCB.getvarallsta(var='V m/s', by='H', From=From, To = To ).resample('H').mean().mean(axis=1)
    
 
    df_T_D = net_LCB.getvarallsta(var='Ta C', by='H', From=From, To = To ).resample('D').mean().mean(axis=1)
    df_Q_D = net_LCB.getvarallsta(var='Ua g/kg', by='H', From=From, To = To ).resample('D').mean().mean(axis=1)
    df_U_D = net_LCB.getvarallsta(var='U m/s', by='H', From=From, To = To ).resample('D').mean().mean(axis=1)
    df_V_D = net_LCB.getvarallsta(var='V m/s', by='H', From=From, To = To ).resample('D').mean().mean(axis=1)
    
    df_cov_H = transport(df_T_H,df_Q_H,df_U_H,df_V_H)
#     df_cov_D = transport(df_T_D,df_Q_D,df_U_D,df_V_D)
#     
#     df_cov_D = df_cov_D.resample('H').fillna(method='pad')
#     
#     
#     df_cov = pd.concat([df_cov_H, df_cov_D], join='outer', axis=1)
    df_cov_H.plot()
    plt.show()
#     plt.plot(df)
#     
#     df_co.plot()
#     plt.show()
















