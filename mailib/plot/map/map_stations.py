#===============================================================================
#    DESCRIPTION 
#        make a beautiful map of the station position
#===============================================================================

# from mpl_toolkits.basemap import Basemap,cm
# from stadata.lib.LCBnet_lib import att_sta
from mailib.stanet.lcbstanet import Att_Sta

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap



def map_domain(raster_lat, raster_lon, raster_val, llcrnrlon = -49, urcrnrlon = -43, llcrnrlat = -25 ,urcrnrlat = -21,clevs=None,cmap=plt.get_cmap('Greys'), ax=None):
    """
    PARAMETERS:
        raster_val: 2d numpy array
        raster_lon:2d numpy array
        raster_val:2d numpy array
    """

    # raster_val = raster_val[::-1,:] # transformation
    # raster_lon = raster_lon[::-1,:] # transformation
    # raster_lat = raster_lat[::-1,:] # transformation
#
#     llcrnrlon = raster_lon.min()
#     urcrnrlon = raster_lon.max()
#     llcrnrlat = raster_lat.min()
#     urcrnrlat = raster_lat.max()
#
    #    Serra Da Mantiquera
    # llcrnrlon = -49
    # urcrnrlon = -43
    # llcrnrlat = -25
    # urcrnrlat = -21

    map = Basemap(projection='mill',lat_ts=10,llcrnrlon=llcrnrlon, \
    urcrnrlon=urcrnrlon,llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat, \
    resolution='i',area_thresh=10000)

    inpath ="/home/thomas/phd/geomap/data/shapefile/"

    # # brazil regions
    # shapefile= inpath + "shapefile_brasil_python/BRA_adm1"
    # map.readshapefile(shapefile, 'BRA_adm1', drawbounds=True, linewidth=1.5, color='#536283')
    #
    # # Cantareira
    # shapefile= inpath+ 'cantareira/cantareiraWGS'
    # map.readshapefile(shapefile, 'cantareiraWGS', drawbounds=True, linewidth=1.5, color='#48a3e6')
    #
    # # RMSP
    # shapefile= inpath+ 'rmsp/rmsp_polig'
    # map.readshapefile(shapefile, 'rmsp_polig', drawbounds=True, linewidth=1.5, color='#890045')

    #===============================================================================
    # Draw paralells
    #===============================================================================

    parallels = np.arange(llcrnrlat, -urcrnrlat,0.1)
    map.drawparallels(parallels,labels=[1,0,0,0],fontsize=11, linewidth=0.2)
    # draw meridians
    meridians = np.arange(llcrnrlon, urcrnrlon,0.1)
    map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=11, linewidth=0.2)

    map.drawmapscale(urcrnrlon-0.1, llcrnrlat+0.1, urcrnrlon+0.1, llcrnrlat+0.1, 50, barstyle='fancy',fontsize=11,units='km')

    #===========================================================================
    # Background raster
    #===========================================================================


    # ny = raster_val.shape[0]; nx = raster_val.shape[1]

#     lons, lats = map.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
#     x, y = map(lons, lats) # compute map proj coordinates.

    raster_lon = np.array(raster_lon.tolist()).astype(np.float64)
    raster_lat = np.array(raster_lat.tolist()).astype(np.float64)
    raster_val = np.array(raster_val.tolist()).astype(np.float64)

    x, y = map(raster_lon, raster_lat) # compute map proj coordinates.




    cs = map.contour(x.astype(np.float64), y.astype(np.float64), raster_val.astype(np.float64))#, levels=clevs, cmap=cmap, ax=ax)
    cbar = map.colorbar(cs,location='bottom',pad="5%", ax=ax)
    cbar.set_label('m')

    plt.legend(loc=1, numpoints=1, framealpha=0.4, fontsize=11)
    map.drawmapboundary()
#     plt.show()
    return plt, map




def map_domain(raster_lat, raster_lon, raster_val, llcrnrlon=None,urcrnrlon=None,llcrnrlat=None, urcrnrlat=None, ax=None, clevs=None, ribeirao=False):
    """
    PARAMETERS:
        raster_val: 2d numpy array
        raster_lon:2d numpy array
        raster_val:2d numpy array
    """


#     raster_val = raster_val[::-1,:] # transformation
#     raster_lon = raster_lon[::-1,:] # transformation
#     raster_lat = raster_lat[::-1,:] # transformation
#
    # if llcrnrlon== None:
    #
    if llcrnrlon == None:
        llcrnrlon = raster_lon.min()
        urcrnrlon = raster_lon.max()
        llcrnrlat = raster_lat.min()
        urcrnrlat = raster_lat.max()


    # fig     = plt.figure()
    # ax      = fig.add_subplot(111)

    map = Basemap(projection='mill',lat_ts=10,llcrnrlon=llcrnrlon, \
    urcrnrlon=urcrnrlon,llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat, \
    resolution='c', ax=ax)

    inpath ="/home/thomas/phd/geomap/data/shapefile/"

    # brazil regions
    # shapefile= inpath + "shapefile_brasil_python/BRA_adm1"
    # map.readshapefile(shapefile, 'BRA_adm1', drawbounds=True, linewidth=0.3, color='#536283', zorder=0)

    # Brazil water
    # shapefile= inpath + "ANA_atlantic_sudeste/Hidrografia 250000"
    # map.readshapefile(shapefile, 'Hidrografia 250000', drawbounds=True, linewidth=0.5, color='#536283')

    # # Cantareira
    # shapefile= inpath+ 'cantareira/cantareiraWGS'
    # map.readshapefile(shapefile, 'cantareiraWGS', drawbounds=True, linewidth=1.5, color='#48a3e6')
    #
    # # RMSP
    # shapefile= inpath+ 'rmsp/rmsp_polig'
    # map.readshapefile(shapefile, 'rmsp_polig', drawbounds=True, linewidth=1.5, color='#890045')

    if ribeirao:
        shapefile= inpath + "shape_Sub-bacias/bacias"
        map.readshapefile(shapefile, 'bacias', drawbounds=True, linewidth=3, color='#536283', zorder=0)
        # map.drawmapscale(urcrnrlon-0.01, llcrnrlat+0.005, urcrnrlon, llcrnrlat, 2, barstyle='fancy',fontsize=11,units='km')

    #===============================================================================
    # Draw paralells
    #===============================================================================
    #
    # parallels = np.arange(llcrnrlat, urcrnrlat,1)
    # map.drawparallels(parallels,labels=[1,0,0,0],fontsize=11, linewidth=0.2)
    # # draw meridians
    # meridians = np.arange(llcrnrlon, urcrnrlon,1)
    # map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=11, linewidth=0.2)

    #===========================================================================
    # Background raster
    #===========================================================================

    if clevs == None:
        clevs = np.linspace(raster_val.min(), raster_val.max(), 20)
    ny = raster_val.shape[0]; nx = raster_val.shape[1]

#     lons, lats = map.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
#     x, y = map(lons, lats) # compute map proj coordinates.

    print('start map')
    x, y = map(raster_lon, raster_lat) # compute map proj coordinates.

    print('contour')
    cs = map.contourf(x, y, raster_val, levels=clevs, cmap=plt.get_cmap('Greys'), alpha=0.8, extend='both')

    # plt.legend(loc=1, numpoints=1, framealpha=0.4, fontsize=11)
    map.drawmapboundary()
    # map.drawcoastlines()
#     plt.show()
    return plt, map

def add_stations_positions(map, stalatlon,s=60):
    
    #===============================================================================
    # Stations position
    #===============================================================================
    marker= iter(['^', 's', 'v', 'o','o', 'o', 'D','p','d'])
    color= iter(['#2171b5', '#eb0096', '#990099','#6baed6', '#7086f7', '#aa6c91',"#ff8000", "#99e699","#ffff66"])
    for net in ['ribeirao']:
        lat = stalatlon[stalatlon.loc[:,'network'] == net].loc[:,'lat'].values
        lon = stalatlon[stalatlon.loc[:,'network'] == net].loc[:,'lon'].values
        m = next(marker)
        c = next(color)
        x , y = map(lon, lat)
        map.plot(x,y , c=c, marker=m, linestyle='', markersize=s, label=net)
#         map.scatter(x, y, marker=m, color='k',  label=net)


    return map

def add_wind_vector(map, var_u, var_v, stalatlon, scale=1.25,linewidth=0.01, width=0.01, scale_quiverkey=0.1, quiver_legend=r'$0.1$', colorvector='k',alphavector=None, xkey=0.9, ykey=1.05):
    """
    Draw wind vector on basemap
    """

    lat = stalatlon.loc[:,'Lat'].values
    lon = stalatlon.loc[:,'Lon'].values
    x , y = map(lon, lat)
    q = map.quiver(x, y, var_u, var_v, linewidth=linewidth, width=width, scale=scale, color=colorvector,alpha=alphavector)
    qk = plt.quiverkey(q, xkey, ykey, scale_quiverkey,quiver_legend , labelpos='E',
                   coordinates='axes', alpha=0.5)
    return map

def add_loadings_as_marker(map, loadings, stalatlon, vmin=None, vmax=None):
    """
    values: pandas serie
    """
    lat = stalatlon.loc[:, 'Lat'].values
    lon = stalatlon.loc[:,'Lon'].values
#     scaled_values = minmax_scale(loadings.values)
    scaled_values = loadings.values
    colors = plt.cm.RdBu(scaled_values)
    x , y = map(lon, lat)
    cmap = plt.get_cmap('RdBu_r')#, 10)
    sc = map.scatter(x,y , marker='o', c=scaled_values, cmap=cmap, s=40, vmin=vmin, vmax=vmax)
    # plt.colorbar(sc)
    return map, sc

if __name__=='__main__':
    # Plot map Ribeirao
    raster_val = np.loadtxt("/vol0/thomas.martin/test/2124_4448latitude.txt", delimiter=',')
    raster_lon = np.loadtxt("/vol0/thomas.martin/test/2124_4448longitude.txt", delimiter=',')
    raster_lat = np.loadtxt("/vol0/thomas.martin/test/2124_4448@PERMANENT", delimiter=',')

    att_sta = Att_Sta('/vol0/thomas.martin/framework/database/stations_database/1_database/metadata/database_metadata.csv')
    # AttSta = att_sta()
    stalatlon = att_sta.attributes.loc[:, ['lat','lon','network']]
    stalatlon = stalatlon.filter(like='rib',axis=0)

    lat_min = -23.65
    lat_max = -23.45
    lon_min = -46.85
    lon_max = -46.65
    print('start map domain')
    plt, map = map_domain(raster_lat, raster_lon, raster_val, llcrnrlon=lon_min,urcrnrlon=lon_max,llcrnrlat=lat_min, urcrnrlat=lat_max)
    print('add stations')
    map = add_stations_positions(map,stalatlon)
    print('legend')
    plt.legend(loc='best', framealpha=0.4)
    
    # plt.show()
    print('plot')
    plt.savefig("/vol0/thomas.martin/map_stations_domain_regional.pdf", transparent=True, dpi=500 )

        
        

        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
