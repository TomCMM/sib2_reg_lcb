
# from statmod_lib import *
from clima_lib.LCBnet_lib import *
# from mapstations import plot_local_stations
from numpy.testing.utils import measure
import datetime 
import matplotlib
import sys
import pickle
import pandas as pd
import arps_lib
from  arps_lib.plot_map_basemap import beautiful_map
if __name__=='__main__':
 

#     loadings_sta = pickle.load( open( "save.p", "rb" ))
#     print 'aaaa'
#     print loadings_sta

    #===============================================================================
    # Modis at station points
    #===============================================================================
    
    AttSta = att_sta()
    stanames  =AttSta.stations(all=True)
     
    df_lons = pd.read_csv('/home/thomas/phd/statmod/data/modis_data/modis_lons_points.csv',index_col=0, parse_dates=True)
    df_lats = pd.read_csv('/home/thomas/phd/statmod/data/modis_data/modis_lats_points.csv',index_col=0, parse_dates=True)
    df_lats = df_lats.dropna(axis=0,how='all')
    df_lons = df_lons.dropna(axis=0,how='all')
    df_lats.index = df_lats.index.round('H')
    df_lons.index = df_lons.index.round('H')
     
      
    df_modis = pd.read_csv('/home/thomas/phd/statmod/data/modis_data/modis_data_points.csv',index_col=0, parse_dates=True)
    df_modis.columns = stanames
#     print df_lats
#     df_modis = df_modis.loc[:,df_lons.columns]
     
    df_modis.index = df_modis.index.round('H')
    df_modis = df_modis.dropna(axis=0,how='all')
    df_modis[df_modis ==0]=np.nan
    df_modis = df_modis * 0.02 -273.15
 
    idx_bad_lat = df_lats[df_lats.mean(axis=1).diff().abs() <0.005].index
    idx_bad_lon = df_lons[df_lons.mean(axis=1).diff().abs() <0.005].index

  
    df_lats = df_lats.loc[idx_bad_lat,:]
    df_lats = df_lats.loc[idx_bad_lon,:]
    df_lons = df_lons.loc[df_lats.index,:]
    df_lons = df_lons.loc[df_lons.index,:]
    df_modis = df_modis.loc[df_lats.index,:]
#     

#     AttSta = att_sta(Path_att="/home/thomas/PhD/obs-lcb/staClim/metadata_allnet_select.csv")
#     stanames  =AttSta.stations(all=True)
    
    df_lons_map = pd.read_csv('/home/thomas/phd/statmod/data/modis_data/modis_lons_map.csv',index_col=0, parse_dates=True)
    df_lats_map = pd.read_csv('/home/thomas/phd/statmod/data/modis_data/modis_lats_map.csv',index_col=0, parse_dates=True)
    df_lats_map = df_lats_map.dropna(axis=0,how='all')
    df_lons_map = df_lons_map.dropna(axis=0,how='all')
    df_lats_map.index = df_lats_map.index.round('H')
    df_lons_map.index = df_lons_map.index.round('H')

    
    df_modis_map = pd.read_csv('/home/thomas/phd/statmod/data/modis_data/modis_data_map.csv',index_col=0, parse_dates=True)
#     df_modis_map.columns = stanames
#     df_modis_map = df_modis_map.loc[:,loadings_sta.columns]
     
    df_modis_map.index = df_modis_map.index.round('H')
    df_modis_map = df_modis_map.dropna(axis=0,how='all')
    df_modis_map[df_modis_map ==0]=np.nan
    df_modis_map = df_modis_map * 0.02 -273.15
    
    
    
    df_modis = df_modis_map
    
#     print df_modis
 
    idx_bad_lat = df_lats_map[df_lats_map.mean(axis=1).diff().abs() <0.005].index
    idx_bad_lon = df_lons_map[df_lons_map.mean(axis=1).diff().abs() <0.005].index
   
    df_lats_map = df_lats_map.loc[idx_bad_lat,:]
    df_lats_map = df_lats_map.loc[idx_bad_lon,:]
    df_lons_map = df_lons_map.loc[df_lats.index,:]
    df_lats_map = df_lats_map.loc[df_lons.index,:]
    df_modis_map = df_modis_map.loc[df_lats.index,:]  
     

#     print df_modis.shape
#     df_modis_map = pd.concat([df_modis, df_modis_map], axis=1, join ='inner')
#     print df_modis.shape
#     df_lons = pd.concat([df_lons, df_lons_map], axis=1, join ='inner')
#     df_lats = pd.concat([df_lats, df_lats_map], axis=1, join ='inner')
# 
#     print df_modis

#     df_modis = df_modis.iloc[:,::4]
#     
#     df_modis= df_modis.between_time('08:00', '17:00')
    df_modis= df_modis.between_time('16:00', '07:00')
#     df_modis = df_modis[(df_modis.index.month <11) & (df_modis.index.month >3)]
    
   
#     df_modis = df_modis.loc[:,df.columns]
#   
#     df_modis.plot()
#    
  
    #===========================================================================
    # TEST PCA
    #===========================================================================
    from sklearn.decomposition import PCA
        
    nbpc = 5
    nbpts = 200
    pca_modis = PCA(n_components=nbpc)
     
    print df_modis.shape
    print df_modis.mean(axis=1)
    df_modis = df_modis.subtract(df_modis.mean(axis=1), axis='index')
#     df_modis = (df_modis - df_modis.mean(axis=0)) / df_modis.std(axis=0)
         
#     print df_modis.max().max()
#     print df_modis.min().min()
         
         
    df_modis = df_modis.interpolate(axis=0)
         
         
         
# 
#     df_modis = df_modis.fillna(df_modis.mean())
          
    df_modis.dropna(axis=0, inplace=True, how='any')
    print df_modis.shape
#     from sklearn.preprocessing import StandardScaler
#     df_modis = StandardScaler().fit_transform(df_modis)    
#     print df_modis.shape
       
       
    df_modis_scores = pca_modis.fit_transform(df_modis)
       
# 
#     # plot cuulated variance
#     eig_vals = pca_modis.explained_variance_
#     tot = sum(eig_vals)
#     var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
#     cum_var_exp = np.cumsum(var_exp)
#         
#     plt.bar(range(len(var_exp)),var_exp)
#     plt.ylabel("Explained variance")
#     plt.xlabel('PCs')
#     plt.show()
# # #      
#     plt.plot(df_modis.index,df_modis_scores, linewidth=3)
#     plt.axhline(0,linewidth=5,c='0.5')
#     plt.show()
       
#     data=pd.DataFrame(df_modis_scores, index=df_modis).groupby(lambda t: (t.hour)).mean()
#     print data
#     plt.plot(data)
#     plt.show()
   
    loadings = pca_modis.components_
    print loadings
    print loadings.shape
#     for pc in range(nbpc -1):
#         plt.scatter(loadings[pc,:],loadingsmap[pc,:55], alpha=0.10)
#         plt.xlabel("PC sta "+str(pc+1))
#         plt.ylabel("PC sta taken with map"+str(pc+1))
#         plt.show()
#     
    
    
#     
# 
# #     for i in df_modis.index:
# #         d = np.reshape(df_modis.loc[i,:] , (100,100))
# #         lvls = np.linspace(d.min(), d.max(),100)
# #         plt.contourf(d, levels = lvls, cmaps='bwr')
# #         plt.colorbar()
# #         plt.show()
# 
# 
#     for pc in range(nbpc -1):
#         plt.scatter(loadings[pc,:],loadings[pc+1,:], alpha=0.10)
#         plt.xlabel("PC"+str(pc+1))
#         plt.ylabel("PC"+str(pc+2))
#         plt.show()
# #     
# #     

    for pc in range(nbpc):
        var = np.reshape(loadings[pc,:] , (nbpts,nbpts))
        lat = np.reshape(df_lats_map.iloc[0,:], (nbpts,nbpts))
        lon = np.reshape(df_lons_map.iloc[0,:], (nbpts,nbpts))
        plt = beautiful_map(var, lat, lon, varunits=str(pc+1)+" loading", title = "Map of the PC " + str(pc+1) + ' loading')
        plt.show()
#     for pc in range(nbpc):
#         plt.figure()
#         plt.title("Loadings PC"+str(pc+1))
# #         print loadings[pc,55:].shape
#         l1 = np.reshape(loadings[pc,:] , (nbpts,nbpts))
#         lvls = np.linspace(l1.min(), l1.max(),100)
#         beautiful_map()
#         plt.contourf(l1, levels = lvls, cmap="bwr")
# #         plt.imshow(l1, interpolation="nearest")
#         plt.xlabel("PC"+str(pc+1))
#         plt.ylabel("PC"+str(pc+2))
#         plt.colorbar()
#         plt.show()
# #         
#        
#        
#         from mpl_toolkits.basemap import Basemap,cm
#               
#         llcrnrlon = -47.6
#         urcrnrlon = -45.6
#         llcrnrlat = -23.6
#         urcrnrlat = -21.6
#                
#         m=Basemap(projection='mill',lat_ts=10,llcrnrlon=llcrnrlon, \
#         urcrnrlon=urcrnrlon,llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat, \
#         resolution='i',area_thresh=10000)
#            
#         shapefile='/home/thomas/PhD/obs-lcb/staClim/box/loc_sta/shapefile_brasil_python/BRA_adm1'
#         m.readshapefile(shapefile, 'BRA_adm1', drawbounds=True)
#        
#         shapefile='/home/thomas/PhD/obs-lcb/staClim/box/loc_sta/shape_Sub-bacias/bacias'
#         m.readshapefile(shapefile, 'bacias', drawbounds=True)
#        
#        
#         lats = np.reshape(df_lats_map.iloc[0,:] , (nbpts,nbpts))
#         lons = np.reshape(df_lons_map.iloc[0,:] , (nbpts,nbpts))
#            
#            
#         x, y = m(lons, lats) # compute map proj coordinates.
#    
# #         clevs = range(int(l1.min()), int(l1.max()))
#         cs = m.contourf(x,y,l1,lvls, cmap=plt.get_cmap('gist_ncar'))
#         cbar = m.colorbar(cs,location='bottom',pad="5%")
#         cbar.set_label('m')
#         plt.show()
# # #         
# #   
# #     print loadings
#            
# #     loadings = pd.DataFrame(loadings[:55,:], index =  loadings_sta.index, columns=loadings_sta.columns)
#    
#     print loadings.shape
#     loadings = pd.DataFrame(loadings[:,:55], index = [str(i)+"sat"for i in loadings_sta.index], columns=loadings_sta.columns)
#        
#     loadings_sta = loadings_sta.T
#     loadings = loadings.T
#     
#     print loadings_sta
#     print loadings
#      
#      
#     df = pd.concat([loadings,loadings_sta],axis=1,join='inner')
#     print df
#     from pandas.tools.plotting import scatter_matrix
#     scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
#     plt.show()
#     
#     
#     
#     
#     for i in range(5):
#         fig, ax = plt.subplots()
#         plt.scatter(loadings_sta.values[i,:],loadings.values[i,:] )
#         for j,name in enumerate(loadings_sta.columns):
#             ax.annotate(name, (loadings_sta.values[i,j],loadings.values[i,j] ))
#         plt.show()
#          
#     fig, ax = plt.subplots()
#     plt.scatter(loadings_sta.values[3,:],loadings.values[2,:] )
#     for j,name in enumerate(loadings_sta.columns):
#         print name
#         ax.annotate(name, (loadings_sta.values[3,j],loadings.values[2,j] ))
#     plt.xlabel('PC4 stations')
#     plt.ylabel('PC3 satelite')
#     plt.show()
#        
#        
#     print loadings_sta
#     print loadings
#        
#     fig, ax = plt.subplots()
#     plt.scatter(loadings_sta.values[3,:],loadings.values[3,:] )
#     for j,name in enumerate(loadings_sta.columns):
#         print name
#         ax.annotate(name, (loadings_sta.values[3,j],loadings.values[3,j] ))
#     plt.xlabel('PC4 stations')
#     plt.ylabel('PC4 satelite')
#     plt.show()
#       
# 
# #      
# # 
#      
#      
#      
#     
#      
# 
#     