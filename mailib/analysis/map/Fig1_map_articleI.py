from map_stations import *

if __name__=='__main__':
    raster_val = np.loadtxt("/home/thomas/phd/geomap/data/regional_raster/map_lon44_49_lat20_25_lowres", delimiter=',')
    raster_lon = np.loadtxt("/home/thomas/phd/geomap/data/regional_raster/map_lon44_49_lat20_25_lowreslongitude.txt", delimiter=',')
    raster_lat = np.loadtxt("/home/thomas/phd/geomap/data/regional_raster/map_lon44_49_lat20_25_lowreslatitude.txt", delimiter=',')

    AttSta = att_sta('/home/thomas/phd/obs/staClim/metadata/database_metadata.csv')
    stalatlon = AttSta.attributes.loc[:, ['Lat','Lon','network']]
    print(stalatlon)
    stalatlon = stalatlon[stalatlon['network']=='ribeirao']

    plt, map = map_domain(raster_lat, raster_lon, raster_val, llcrnrlon = -46.40, urcrnrlon = -46.20, llcrnrlat = -22.9 ,urcrnrlat = -22.7)
    map = add_stations_positions(map,stalatlon)
    plt.legend(loc='best', framealpha=0.4)

    plt.show()
    # plt.savefig("/home/thomas/phd/climaobs/res/map/map_stations_domain_regional.eps", transparent=True )
