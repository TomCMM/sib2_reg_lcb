#===============================================================================
# DESCRIPTION
#    Make an analysis of the wind at the regional scale
#===============================================================================

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot


if __name__ == "__main__":
    #===============================================================================
    # Get the Data
    #===============================================================================
    Lat = [-25,-21]
    Lon = [-48, -44]
    Alt = [500,5000]
    
    net_sinda = LCB_net()
    net_inmet = LCB_net()
    net_iac = LCB_net()
    net_LCB = LCB_net()
    net_svg =  LCB_net()
    net_peg =  LCB_net()
                   
#     Path_Sinda = '/home/thomas/PhD/obs-lcb/staClim/Sinda/obs_clean/Sinda/'
    Path_INMET ='/home/thomas/phd/obs/staClim/inmet/full/'
    Path_IAC ='/home/thomas/phd/obs/staClim/iac/data/full/'
    Path_LCB='/home/thomas/phd/obs/lcbdata/obs/full_sortcorr/'
    Path_svg='/home/thomas/phd/obs/staClim/svg/SVG_2013_2016_Thomas_30m.csv'
    Path_peg='/home/thomas/phd/obs/staClim/peg/Th_peg_tar30m.csv'
      
    AttSta_IAC = att_sta('/home/thomas/phd/local_downscaling/data/metadata/df_topoindex_regional.csv')
    AttSta_IAC.attributes.dropna(how='all',axis=1, inplace=True)
    AttSta_IAC.attributes.dropna(how='any',axis=0, inplace=True)       
      
    AttSta_Inmet = att_sta('/home/thomas/phd/local_downscaling/data/metadata/df_topoindex_regional.csv')
    AttSta_Inmet.attributes.dropna(how='all',axis=1, inplace=True)
    AttSta_Inmet.attributes.dropna(how='any',axis=0, inplace=True)    
    
#     AttSta_Sinda = att_sta('/home/thomas/phd/local_downscaling/data/topo_indexes/df_topoindex/df_topoindex.csv')
    AttSta_LCB = att_sta('/home/thomas/phd/local_downscaling/data/metadata/df_topoindex_regional.csv')
    AttSta_LCB.attributes.dropna(how='all',axis=1, inplace=True)
    AttSta_LCB.attributes.dropna(how='any',axis=0, inplace=True)
         
    AttSta_IAC.setInPaths(Path_IAC)
    AttSta_Inmet.setInPaths(Path_INMET)
#     AttSta_Sinda.setInPaths(Path_Sinda)
    AttSta_LCB.setInPaths(Path_LCB)
    
    stanames_IAC =  AttSta.get_sta_id_in_metadata(values=['IAC'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt}) # this does not work anymore
    stanames_Inmet = AttSta.get_sta_id_in_metadata(values=['Innmet'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt})
#     stanames_Sinda = AttSta.stations(values=['Sinda'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt} )
    stanames_LCB = AttSta.get_sta_id_in_metadata(values = ['Ribeirao'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt})
#     stanames_LCB = ['C10', 'C08','C15','C05' ]
    [stanames_IAC.remove(x) for x in ['pc58','sb69'] if x in stanames_IAC ] # Remove stations
#     [stanames_LCB.remove(x) for x in ['C10','C17','C12','C14','C08'] if x in stanames_LCB ] # Remove stations
    [stanames_Inmet.remove(x) for x in ['A531'] if x in stanames_Inmet ] # Remove stations 
    
#------------------------------------------------------------------------------ 
# Create Dataframe
#------------------------------------------------------------------------------ 
    Files_IAC =AttSta_IAC.getatt(stanames_IAC,'InPath')
    Files_Inmet =AttSta_Inmet.getatt(stanames_Inmet,'InPath')
#     Files_Sinda =AttSta_Sinda.getatt(stanames_Sinda,'InPath')
    Files_LCB =AttSta_LCB.getatt(stanames_LCB,'InPath')
         
#     net_sinda.AddFilesSta(Files_Sinda, net='Sinda')
    net_inmet.AddFilesSta(Files_Inmet, net='INMET')
    net_iac.AddFilesSta(Files_IAC, net='IAC')

    net_LCB.AddFilesSta(Files_LCB)
    net_svg.AddFilesSta([Path_svg], net='svg')
    net_peg.AddFilesSta([Path_peg], net='peg')
    
    df_iac = net_iac.getvarallsta(var=var,by='H',From=From, To = To)
    df_inmet = net_inmet.getvarallsta(var=var,by='H',From=From, To = To)
#     X_sinda = net_sinda.getvarallsta(var=var,by='3H',From=From, To = To)
    df_LCB = net_LCB.getvarallsta(var=var, by='H', From=From, To = To )
    df_svg = LCB_station(Path_svg, net='svg').getData(var=var, by='H', From=From, To = To )
    df_svg.columns =['svg']
    df_peg = LCB_station(Path_peg, net='peg').getData(var=var, by='H', From=From, To = To )
    df_peg.columns =['peg']
  
    df = pd.concat([ df_LCB, df_iac], axis=1, join='inner')
#     df = df.between_time('19:00','05:00')
#     df = df.resample("D").min()
#     df_gfs = df_gfs.resample('D').min()
#     df = df.T
#     df.plot(legend=False)
#     plt.xlabel('Date')
#     plt.ylabel('Temperture (C)')
#     
#     plt.show()   
     
#     df = df.fillna(df.mean(), axis=0)
    df = df.dropna(axis=0,how='any')