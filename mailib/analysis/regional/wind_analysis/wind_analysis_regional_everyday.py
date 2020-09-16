'''
DESCRIPTION
    Analysis of the wind in the Serra Da mantiquer and Serra Do mar

Created on 28/05/2017

@author: thomas
'''

import numpy as np
import pandas as pd

from mailib.plot.map.map_stations import map_domain, add_stations_positions, add_wind_vector
from stadata.lib.LCBnet_lib import att_sta

# plt.rcParams["figure.figsize"] = (28,28)


if __name__ == '__main__':
    #===============================================================================
    # Read and preprocessing data
    #===============================================================================
    
#     df_sm = pd.read_csv("/home/thomas/phd/climaobs/data/sta_data/Sm.csv", index_col=0, parse_dates=True)
#     df_dm = pd.read_csv("/home/thomas/phd/climaobs/data/sta_data/Dm.csv", index_col=0, parse_dates=True)
#     stanames = df_sm.columns
#     index = df_sm.index
#     
#     df_u = []
#     df_v = []
#     for norm,theta in zip(df_sm.values, df_dm.values):
#         U,V = PolarToCartesian(norm,theta)
#         df_u.append(U)
#         df_v.append(V)
#     df_u = pd.DataFrame(df_u, index=index, columns =stanames).resample('H').mean()
#     df_v = pd.DataFrame(df_v, index=index, columns =stanames).resample('H').mean()

    # df_u = pd.read_csv("/home/thomas/phd/framework/fill/out/Uw.csv", index_col=0, parse_dates=True).resample('H').mean()
    # df_v = pd.read_csv("/home/thomas/phd/framework/fill/out/Vw.csv", index_col=0, parse_dates=True).resample('H').mean()

    df_u = pd.read_csv("/home/thomas/phd/framework/qa/out/qa_visual/Uw.csv", index_col=0, parse_dates=True).resample('H').mean()
    df_v = pd.read_csv("/home/thomas/phd/framework/qa/out/qa_visual/Vw.csv", index_col=0, parse_dates=True).resample('H').mean()

    df_u = df_u.loc['2013-01-01 01:00:00':]
    df_v = df_v.loc['2013-01-01 01:00:00':]

    stanames = df_u.columns
    # group 
    # df_u_daily = df_u.groupby(lambda x: x.hour).mean()
    # df_v_daily = df_v.groupby(lambda x: x.hour).mean()

    # df_u = apply_func(df_u, normalize)
    # df_v = apply_func(df_v, normalize)

    #===========================================================================
    # Plot daily wind vector over the region 
    #===========================================================================

    raster_val = np.loadtxt("/home/thomas/phd/geomap/data/raster/regional_raster/map_lon44_49_lat20_25_lowres", delimiter=',')
    raster_lon = np.loadtxt("/home/thomas/phd/geomap/data/raster/regional_raster/map_lon44_49_lat20_25_lowreslongitude.txt", delimiter=',')
    raster_lat = np.loadtxt("/home/thomas/phd/geomap/data/raster/regional_raster/map_lon44_49_lat20_25_lowreslatitude.txt", delimiter=',')
    
    AttSta = att_sta("/home/thomas/phd/obs/staClim/metadata/database_metadata.csv")
    print AttSta.attributes
    stalatlon = AttSta.attributes.loc[stanames, ['Lat','Lon','network']]
    print stalatlon


    for i, hour in enumerate(df_u.index):
        plt, map = map_domain(raster_lat, raster_lon, raster_val)
        map = add_stations_positions(map,stalatlon)
        map = add_wind_vector(map, df_u.loc[hour,:], df_v.loc[hour,:], stalatlon, scale=15,linewidth=0.001, width=0.0025)
        plt.title(str(hour)+" LT", fontsize=20)
#         plt.legend(loc='best', framealpha=0.4)
#         plt.show()
        outfilename = "/home/thomas/phd/framework/res/regional/wind_every_hour/"+str(hour).rjust(2,'0')+"map_wind__regional.png"
        print outfilename
        plt.savefig(outfilename, dpi=400 )
        plt.close('all')
