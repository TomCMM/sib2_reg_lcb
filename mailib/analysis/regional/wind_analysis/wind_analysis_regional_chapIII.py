'''
DESCRIPTION
    Analysis of the wind in the Serra Da mantiquer and Serra Do mar

Created on 28/05/2017

@author: thomas
'''

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mailib.plot.map.map_stations import map_domain, add_wind_vector
from stadata.lib.LCBnet_lib import att_sta

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["figure.figsize"] = (10,14)
matplotlib.rcParams.update({'font.size': 11})


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



    # df_u = apply_func(df_u, normalize)
    # df_v = apply_func(df_v, normalize)

    stanames = df_u.columns

    # Drop ribeirao stations
    stanames  = stanames.drop(['C08', 'C07','C06', 'C05','C04', 'C10','C11', 'C12','C13', 'C14','C15', 'C16', 'C17','C18','C19'])
    df_u = df_u.loc[:, stanames]
    df_v = df_v.loc[:, stanames]


    # group 
    df_u_daily = df_u.groupby(lambda x: x.hour).mean()
    df_v_daily = df_v.groupby(lambda x: x.hour).mean()

    # # substract mean domain
    # df_u_daily = df_u_daily.sub(df_u_daily.mean(1),axis=0)
    # df_v_daily = df_v_daily.sub(df_v_daily.mean(1),axis=0)

    #===========================================================================
    # Plot daily wind vector over the region 
    #===========================================================================

    raster_val = np.loadtxt("/home/thomas/phd/framework/predictors/in/topo/regional/4943_2125_lowres@PERMANENT", delimiter=',')
    raster_lon = np.loadtxt("/home/thomas/phd/framework/predictors/in/topo/regional/4943_2125_lowreslongitude.txt", delimiter=',')
    raster_lat = np.loadtxt("/home/thomas/phd/framework/predictors/in/topo/regional/4943_2125_lowreslatitude.txt", delimiter=',')
    
    AttSta = att_sta("/home/thomas/phd/obs/staClim/metadata/database_metadata.csv")
    print AttSta.attributes
    stalatlon = AttSta.attributes.loc[stanames, ['Lat','Lon','network']]




    llcrnrlon = -49
    urcrnrlon = -43
    llcrnrlat = -25
    urcrnrlat = -21

    print stalatlon

    gs = gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    ax5 = plt.subplot(gs[2, 0])
    # ax6 = plt.subplot(gs[5, 0])

    clevs = np.linspace(200, 1800, 17)
    plt, map, cs = map_domain(raster_lat, raster_lon, raster_val, ax=ax1, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, clevs=clevs)
    # map = add_stations_positions(map,stalatlon)
    map = add_wind_vector(map, df_u_daily.loc[10,:], df_v_daily.loc[10,:], stalatlon, scale=15, linewidth=0.0025, width=0.005, scale_quiverkey=2, quiver_legend=r'$2m.s^{-1}$', colorvector='b')

    plt, map, cs = map_domain(raster_lat, raster_lon, raster_val, ax=ax2, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, clevs=clevs)
    # map = add_stations_positions(map,stalatlon)
    map = add_wind_vector(map, df_u_daily.loc[13,:], df_v_daily.loc[13,:], stalatlon, scale=15, linewidth=0.0025, width=0.005, scale_quiverkey=2, quiver_legend=r'$2m.s^{-1}$', colorvector='b')


    plt, map, cs = map_domain(raster_lat, raster_lon, raster_val, ax=ax3, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, clevs=clevs)
    # map = add_stations_positions(map,stalatlon)
    map = add_wind_vector(map, df_u_daily.loc[15,:], df_v_daily.loc[15,:], stalatlon, scale=15, linewidth=0.0025, width=0.005, scale_quiverkey=2, quiver_legend=r'$2m.s^{-1}$', colorvector='b')

    plt, map, cs = map_domain(raster_lat, raster_lon, raster_val, ax=ax4, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, clevs=clevs)
    # map = add_stations_positions(map,stalatlon)
    map = add_wind_vector(map, df_u_daily.loc[21,:], df_v_daily.loc[21,:], stalatlon, scale=15, linewidth=0.0025, width=0.005, scale_quiverkey=2, quiver_legend=r'$2m.s^{-1}$', colorvector='b')

    plt, map, cs = map_domain(raster_lat, raster_lon, raster_val, ax=ax5, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, clevs=clevs)
    # map = add_stations_positions(map,stalatlon)
    map = add_wind_vector(map, df_u_daily.loc[3,:], df_v_daily.loc[3,:], stalatlon, scale=15, linewidth=0.0025, width=0.005, scale_quiverkey=2, quiver_legend=r'$2m.s^{-1}$', colorvector='b')


    ax1.set_title('10H')
    ax2.set_title('13H')
    ax3.set_title('15H')
    ax4.set_title('21H')
    ax5.set_title('3H')


    # plt.title(str(hour)+" LT", fontsize=20)
#         plt.legend(loc='best', framealpha=0.4)

    for t, ax in zip(["a)","b)",'c)','d)','e)', 'f)', 'g)','h)','i)','j)'],[ax1, ax2,ax3, ax4, ax5]):
        ax.text(-0.05, 1.05, t, transform=ax.transAxes, size=14, weight='bold')

    plt.tight_layout(pad=3)
#         plt.show()
    outfilename = "/home/thomas/phd/framework/res/regional/chapIII/Fig7_wind.png"
#     print outfilename
    plt.savefig(outfilename )
    plt.show()
#     plt.close('all')





######################################### Old graph
#
#
#
#
#     for i, hour in enumerate(df_u_daily.index):
#         plt, map, cs = map_domain(raster_lat, raster_lon, raster_val)
#         map = add_stations_positions(map,stalatlon)
#         map = add_wind_vector(map, df_u_daily.loc[hour,:], df_v_daily.loc[hour,:], stalatlon, scale=15, linewidth=0.001, width=0.0025)
#         plt.title(str(hour)+" LT", fontsize=20)
# #         plt.legend(loc='best', framealpha=0.4)
# #         plt.show()
#         outfilename = "/home/thomas/phd/framework/res/regional/wind/daily_wind/"+str(i).rjust(2,'0')+"map_wind__regional.png"
#         print outfilename
#         plt.savefig(outfilename, dpi=400 )
#         plt.close('all')
