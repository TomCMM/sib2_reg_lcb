#===============================================================================
# DESCRIPTION
#    The objective of this script is to perform the climatic variabilty along a line 
#    It focus on the variability of the temperature on the axes Serra Da Mantiquera, Serra do Mar
#===============================================================================


# from statmod_lib import *
from LCBnet_lib import *
from staclim.mapstations import plot_local_stations
from numpy.testing.utils import measure
import datetime 
import matplotlib
from mpl_toolkits.basemap import Basemap,cm
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 24}

matplotlib.rc('font', **font)

if __name__=='__main__':
 
#     topo_val  = np.loadtxt('/home/thomas/ASTGTM2_S23W047_dem@PERMANENT',delimiter=',')
#     topo_val = topo_val[::-1,:]
#     topo_lat  = np.loadtxt('/home/thomas/latitude.txt',delimiter=',')
#     topo_lon  = np.loadtxt('/home/thomas/longitude.txt',delimiter=',')
 
    llcrnrlon = -49
    urcrnrlon = -44.5
    llcrnrlat = -24.5
    urcrnrlat = -20.5
 
    Lat = [llcrnrlat,urcrnrlat]
    Lon = [llcrnrlon, urcrnrlon]
    Alt = [400,5000]

    #===========================================================================
    # Load shapefile
    #===========================================================================
    m=Basemap(projection='mill',lat_ts=10,llcrnrlon=llcrnrlon, \
    urcrnrlon=urcrnrlon,llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat, \
    resolution='i',area_thresh=10000)
#  
#     ny = topo_val.shape[0]; nx = topo_val.shape[1]
#     lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
#     x, y = m(lons, lats) # compute map proj coordinates.
# 
#     plt.close('all')

    parallels = np.arange(llcrnrlat,urcrnrlat,0.25)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10, linewidth=0.2)
    # draw meridians
    meridians = np.arange(llcrnrlon, urcrnrlon,0.25)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10, linewidth=0.2) 

    clevs = np.arange(750, 1550, 50)

    
#     im = m.imshow(topo_val,cmap=plt.get_cmap('BrBG_r'))
    
#     cbar = m.colorbar(im,location='bottom',pad="5%")
#     cbar.set_label('m')
#     
#     
#     cs = m.contour(x,y,topo_val, levels = clevs, colors='0.4', linewidths=0.5)

    
#     m.drawmapscale(-46.22, -22.89, -46.22, -22.89,1000, barstyle='fancy',fontsize=10,units='m')
    # cbar.add_lines(cs)
 
#     shapefile='/home/thomas/PhD/obs-lcb/staClim/box/loc_sta/transect_peg_svg/limite_estudo'
#     m.readshapefile(shapefile, 'limite_estudo', drawbounds=True, linewidth=3, color='0.4')
    shapefile='/home/thomas/PhD/obs-lcb/staClim/box/loc_sta/transect_peg_svg/Grande_transecto'
    m.readshapefile(shapefile, 'Grande_transecto', drawbounds=True, linewidth=3, color='0.4')
    shapefile='/home/thomas/PhD/obs-lcb/staClim/box/loc_sta/shapefile_posses/basin'
    m.readshapefile(shapefile, 'basin', drawbounds=True, linewidth=3, color='0.4')
    shapefile='/home/thomas/PhD/obs-lcb/staClim/box/loc_sta/shapefile_brasil_python/BRA_adm1'
    m.readshapefile(shapefile, 'BRA_adm1', drawbounds=True)
    
    AttSta = att_sta(Path_att="/home/thomas/PhD/obs-lcb/staClim/metadata_allnet_select.csv")
    stanames  =AttSta.get_sta_id_in_metadata(all=True)
#     print stanames.tolist()
    x , y = m(AttSta.getatt(stanames, 'Lon'), AttSta.getatt(stanames, 'Lat'))
    
    
    df_att = pd.DataFrame({'Lon':AttSta.getatt(stanames, 'Lon'), "Lat":AttSta.getatt(stanames, 'Lat'), "Alt":AttSta.getatt(stanames, 'Alt'),"x":x,"y":y, "Nome":AttSta.getatt(stanames, 'Nome')}, index = stanames.tolist())
    
    m.plot(x,y , marker='o',linestyle='', markersize=5, label='stations')
    plt.legend()
    for x_v, y_v, m_v in zip (x,y,stanames):
                plt.annotate(m_v, xy=(x_v,y_v), fontsize=10)
    

    plt.show()

    #===========================================================================
    # Select point in shape
    #===========================================================================
    import fiona
    import shapely.geometry as geo
    from shapely.geometry import LineString, Point
    
    with fiona.open("/home/thomas/PhD/obs-lcb/staClim/box/loc_sta/transect_peg_svg/Grande_transecto.shp") as fiona_collection: # open the collection of shape
    
        # In this case, we'll assume the shapefile only has one record/layer (e.g., the shapefile
        # is just for the borders of a single country, etc.).
        shapefile_record = fiona_collection.next() # get the first shape of the collection
    
        # Use Shapely to create the polygon
        shape =  geo.asShape( shapefile_record['geometry'] )
        points = [geo.Point(lon,lat ) for lon, lat in zip(df_att['Lon'],df_att['Lat'])] # get the latitude/longitude from the attribute dataframe and convert into shapely points
        
        selection = []
        for staname, point in zip(stanames, points):
            if shape.contains(point): # select the station whithin the shape
                selection.append(staname)
    
    df_att_select = df_att.loc[selection, :] # select the attributed of the station in the shape
    print df_att_select
    
    #===========================================================================
    # Project the station position on a line
    #===========================================================================
#     left_line = (-48.25,-21.3) # starting node
#     right_line = (-45,-23.75)
    
    left_line = m(-45,-23.75)
    right_line = m( -48.25,-21.3)
    
    line_transect = LineString([left_line, right_line])
    dist_line = Point(left_line).distance(Point(right_line)) / 1000

    distances_from_starting_node = []
    for staname in df_att_select.index:
        p = Point(df_att_select.loc[staname,'x'], df_att_select.loc[staname,'y'])   
        distances_from_starting_node.append(line_transect.project(p)) # Length along line that is closest to the point. from the starting node
    
    df_att_select['distance']  = np.array(distances_from_starting_node)/1000 # in km
    
    df_att_select['stanames'] = df_att_select.index
    



 
    #===========================================================================
    #  Get Data from selected stations
    #===========================================================================
    net_inmet = LCB_net()
    net_iac = LCB_net()
    net_LCB = LCB_net()
    net_svg =  LCB_net()
    net_peg =  LCB_net()
                 
    Path_INMET ='/home/thomas/PhD/obs-lcb/staClim/INMET-Master/full/'
    Path_IAC ='/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/full_2013_2016/'
    Path_LCB='/home/thomas/PhD/obs-lcb/LCBData/obs/Full_sortcorr/'
    Path_svg='/home/thomas/PhD/obs-lcb/staClim/svg/SVG_2013_2016_Thomas_30m.csv'
    Path_peg='/home/thomas/PhD/obs-lcb/staClim/peg/Th_peg_tar30m.csv'
     
    AttSta_IAC = att_sta()
    AttSta_Inmet = att_sta()
    AttSta_LCB = att_sta()
        
    AttSta_IAC.setInPaths(Path_IAC)
    AttSta_Inmet.setInPaths(Path_INMET)
    AttSta_LCB.setInPaths(Path_LCB)
     
    stanames_IAC = [x for x in AttSta.get_sta_id_in_metadata(values=['IAC']) if x in df_att_select['stanames']]
    stanames_Inmet = [x for x in AttSta.get_sta_id_in_metadata(values=['Innmet']) if x in df_att_select['stanames']]
    stanames_LCB =[x for x in AttSta.get_sta_id_in_metadata(values=['Ribeirao']) if x in df_att_select['stanames']]
     
    print stanames_IAC
    print stanames_Inmet
    print stanames_LCB
    
    Files_IAC =AttSta_IAC.getatt(stanames_IAC,'InPath')
    Files_Inmet =AttSta_Inmet.getatt(stanames_Inmet,'InPath')
    Files_LCB =AttSta_LCB.getatt(stanames_LCB,'InPath')
         
    net_inmet.AddFilesSta(Files_Inmet, net='INMET')
    net_iac.AddFilesSta(Files_IAC, net='IAC')
    net_LCB.AddFilesSta(Files_LCB)
    net_svg.AddFilesSta([Path_svg], net='svg')
    net_peg.AddFilesSta([Path_peg], net='peg')
    
    var = "Ta C"
    From = "2013-01-01 00:00:00"
    To = "2016--01 00:00:00"
    df_iac = net_iac.getvarallsta(var=var,by='H',From=From, To = To)
    df_inmet = net_inmet.getvarallsta(var=var,by='H',From=From, To = To)
#     df_LCB = net_LCB.getvarallsta(var=var, by='H', From=From, To = To )
    df_svg = LCB_station(Path_svg, net='svg').getData(var=var, by='H', From=From, To = To )
    df_svg.columns =['svg']
    df_peg = LCB_station(Path_peg, net='peg').getData(var=var, by='H', From=From, To = To )
    df_peg.columns =['peg']
 
    df = pd.concat([df_iac, df_inmet,df_svg, df_peg ], axis=1,join='outer')
    dist = df_att_select.loc[df.columns, 'distance']
    dist.sort_values(inplace=True)
    print dist
    dist = dist.drop(['bu6','ss76','cg18'])
 
    #===============================================================================
    # Modis transect
    #===============================================================================
    AttSta_modis = att_sta(Path_att="/home/thomas/PhD/obs-lcb/staClim/metadata_allnet_select.csv")
    stanames_modis  =AttSta_modis.get_sta_id_in_metadata(all=True)
    
    df_lons_lines = pd.read_csv('/home/thomas/modis_lons_lines.csv',index_col=0, parse_dates=True)
    df_lats_lines = pd.read_csv('/home/thomas/modis_lats_lines.csv',index_col=0, parse_dates=True)
    df_lats_lines = df_lats_lines.dropna(axis=0,how='all')
    df_lons_lines = df_lons_lines.dropna(axis=0,how='all')
    df_lats_lines.index = df_lats_lines.index.round('H')
    df_lons_lines.index = df_lons_lines.index.round('H')
    
     
     
    df_modis_lines = pd.read_csv('/home/thomas/modis_data_lines.csv',index_col=0, parse_dates=True)
    df_modis_lines.index = df_modis_lines.index.round('H')
    df_modis_lines = df_modis_lines.dropna(axis=0,how='all')
    df_modis_lines[df_modis_lines ==0]=np.nan
    df_modis_lines = df_modis_lines * 0.02 -273.15


    idx_bad_lat_lines = df_lats_lines[df_lats_lines.mean(axis=1).diff().abs() <0.005].index
    idx_bad_lon_lines = df_lons_lines[df_lons_lines.mean(axis=1).diff().abs() <0.005].index
 
    df_lats_lines = df_lats_lines.loc[idx_bad_lat_lines,:]
    df_lats_lines = df_lons_lines.loc[idx_bad_lon_lines,:]
    df_lons_lines = df_lons_lines.loc[df_lats_lines.index,:]
    df_lons_lines = df_lons_lines.loc[df_lats_lines.index,:]
    df_modis_lines = df_modis_lines.loc[df_lats_lines.index,:]
    
    
    
    #===============================================================================
    # Modis at station points
    #===============================================================================
    df_lons = pd.read_csv('/home/thomas/modis_lons_points.csv',index_col=0, parse_dates=True)
    df_lats = pd.read_csv('/home/thomas/modis_lats_points.csv',index_col=0, parse_dates=True)
    df_lats = df_lats.dropna(axis=0,how='all')
    df_lons = df_lons.dropna(axis=0,how='all')
    df_lats.index = df_lats.index.round('H')
    df_lons.index = df_lons.index.round('H')
    
     
    df_modis = pd.read_csv('/home/thomas/modis_data_points.csv',index_col=0, parse_dates=True)
    df_modis.columns = stanames_modis
    df_modis.index = df_modis.index.round('H')
    df_modis = df_modis.dropna(axis=0,how='all')
    df_modis[df_modis ==0]=np.nan
    df_modis = df_modis * 0.02 -273.15

    idx_bad_lat = df_lats[df_lats.mean(axis=1).diff().abs() <0.005].index
    idx_bad_lon = df_lons[df_lons.mean(axis=1).diff().abs() <0.005].index
 
    df_lats = df_lats.loc[idx_bad_lat,:]
    df_lats = df_lons.loc[idx_bad_lon,:]
    df_lons = df_lons.loc[df_lats.index,:]
    df_lons = df_lons.loc[df_lats.index,:]
    df_modis = df_modis.loc[df_lats.index,:]
    
    df_modis = df_modis.loc[:,df.columns]
#     df_modis = df_modis.iloc[:,::-1]#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
#     df_modis.columns = df.columns

    #===========================================================================
    # 
    #===========================================================================
    

    df = df[df.index.isin(df_modis.index)]
    df_modis = df_modis[df_modis.index.isin(df.index)]
    df_modis_lines = df_modis_lines[df_modis_lines.index.isin(df.index)]


#     for sta in df.columns:
#         plt.scatter(df.loc[:,sta],df_modis.loc[:,sta])
#         plt.title(sta)
#         plt.xlabel("Station observations")
#         plt.ylabel("Modis observations")
#         plt.savefig('/home/thomas/'+sta+'.png')


    df_modis_mean  = df_modis.mean(axis=0)
    df_modis_night = df_modis.between_time('18:00', '06:00').mean(axis=0)
    df_modis_day = df_modis.between_time('09:00', '17:00').mean(axis=0)
       
    df_modis_mean_lines  = df_modis_lines.mean(axis=0)
    df_modis_night_lines = df_modis_lines.between_time('18:00', '06:00').mean(axis=0)
    df_modis_day_lines = df_modis_lines.between_time('09:00', '17:00').mean(axis=0)
  
#     
    daily_mean = df.mean(axis=0)
    daily_min = df.between_time('18:00', '06:00').mean(axis=0)
    daily_max = df.between_time('09:00', '17:00').mean(axis=0)



#     #===========================================================================
#     # PLot
#     #===========================================================================
#     fig, ax1 = plt.subplots()
# #     ax1.scatter(dist,mean[dist.index], c='k', s=50)
# #     ax1.plot(dist,mean[dist.index], c='k',linewidth=3, label='Mean stations')
# #              
# #     ax1.scatter(dist,daily_min[dist.index], c='b', s=50)
# #     ax1.plot(dist,daily_min[dist.index], c='b',linewidth=3,  label='Night stations')
# #             
#     
#     ax1.plot(dist,daily_mean[dist.index], c="#3b5998",linewidth=5,  label='Stations')
#     ax1.scatter(dist,daily_mean[dist.index], c="#3b5998", s=80)
#     ax1.set_ylabel(u'Mean temperature ($^{\circ}C$)', color='k')
#     ax1.set_ylim(5,30)
#     ax1.grid()
#      
# #        
# #     for i in df_modis.index:
# #         ax1.plot(np.linspace(dist_line,0,100),df_modis.loc[i,:], c='0.2',linewidth=0.2)
# #   
# #     ax1.plot(np.linspace(dist_line,0,100),df_modis_mean_lines, c='k', linestyle=':',linewidth=3, label='Mean transect Modis')    
# #     ax1.plot(np.linspace(dist_line,0,100),df_modis_night_lines, c='b', linestyle=':',linewidth=3, label='Night transect Modis')
# #     ax1.plot(np.linspace(dist_line,0,100),df_modis_day_lines, c='r', linestyle=':',linewidth=3, label='Day transect Modis')
# #      
#       
#     print df_modis_mean
#     print dist
#       
# #     ax1.plot(dist,df_modis_mean[dist.index], c='k', linestyle='--',linewidth=3, label='Mean point Modis')    
# #     ax1.plot(dist,df_modis_night[dist.index], c='b', linestyle='--',linewidth=3, label='Night point Modis')
#     ax1.plot(dist,df_modis_mean[dist.index], c="#3b5998", linestyle='--',linewidth=5, label='Modis')
#     ax1.scatter(dist,df_modis_mean[dist.index], c="#3b5998", s=80)
#     plt.minorticks_on()
# #     ax1.grid(b=True, which='minor', color='0.5', linestyle=':')
# #     ax1.grid(b=True, which='major', color='k', linestyle='-')
#     
#     for ymaj in ax1.yaxis.get_majorticklocs():
#         ax1.axhline(y=ymaj,ls='-',color='k')
#     for ymin in ax1.yaxis.get_minorticklocs():
#         ax1.axhline(y=ymin,ls=':', color='0.5')   
#     
#     
#     ax1.legend(loc=2)
# #   
# #     for i in df_modis.index:
# #         ax1.plot(dist,df_modis.loc[i,:], c='0.2',linewidth=0.2)
# # #   
# #     ax1.plot(dist,df_modis_mean, c='k', linestyle='--',linewidth=2)    
# #     ax1.plot(dist,df_modis_night, c='b', linestyle='--',linewidth=2)
# #     ax1.plot(dist,df_modis_day, c='r', linestyle='--',linewidth=2)
#     
#     
#     
#     ax2 = ax1.twinx()
#     ax2.fill_between(dist, 0, df_att_select.loc[dist.index, 'Alt'], facecolor= "0.6")
#     ax2.scatter(dist,df_att_select.loc[dist.index, 'Alt'], c = "0.3" , s=50, label='station altitude')
#     ax2.plot(dist,df_att_select.loc[dist.index, 'Alt'], c = "0.3" ,linewidth=0.5)
#        
#    
#     for staname in zip(dist.index):
#         ax2.text(df_att_select.loc[staname,'distance'], df_att_select.loc[staname,'Alt'] +100, df_att_select.loc[staname,'Nome'].values[0], 
#                  rotation=60,horizontalalignment='left',verticalalignment='bottom',fontsize=16, color="0.3")
#     
#     for tl in ax2.get_yticklabels():
#         tl.set_color("0.3")
#    
#        
#     ax2.set_ylim(0,8000)
#     ax2.set_xlim(0,450)
#    
#     ax2.set_ylabel('Stations altitude (m)', color="0.3")
#     ax1.set_xlabel('Distance from the sea (Km)')
# 
#     plt.show()
