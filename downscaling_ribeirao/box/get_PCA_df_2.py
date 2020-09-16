#===============================================================================
# DESCRIPTION
# Perform the PCA and merge tghe necessary varaible to make the downscalling
# Output of a dataframe to make the multilinear regression
#===============================================================================

from statmod_lib import *






if __name__ == '__main__':
    
    var = "Ta C"
    AttSta = att_sta()
    From = "2015-03-01 00:00:00"
    To = "2016-01-01 00:00:00"
    Lat = [-25,-21]
    Lon = [-48, -45]
    Alt = [400,5000]

    #    Irradiance from R.sun accounting for the observed ratio
    #    Altitude from DEM
    #    Note
    #    I am using the elevation from the DEm for ease.
    #    However in the future, I should put the observed altitude 
    # inpath = "/home/thomas/PhD/supmod/PCA_data/"
    # stanames = ['C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19']
    # alt = pd.read_csv(inpath+"Irradiance_rsun_lin2_lag-0.2_elev_df.csv", index_col = 0, parse_dates=True)
    #      
    # irr = LCB_Irr()
    # inpath_obs = '/home/thomas/PhD/obs-lcb/LCBData/obs/Irradiance/data/'
    # files_obs = glob.glob(inpath_obs+"*")
    # irr.read_obs(files_obs)
    #     
    # irr.read_sim(inpath+"Irradiance_rsun_lin2_lag-0.2_glob_df.csv")
    # # this should be implemented in the Grass code directly
    # irr.data_sim.columns = stanames
    # alt.columns = stanames
          
    # rng = pd.date_range(irr.data_sim.index[0], irr.data_sim.index[-1],freq='1H')
    # df_0 = pd.DataFrame(0,index =rng, columns=stanames)
    # 
    # 
    # irr.data_sim = irr.data_sim.reindex(rng)
    # alt = alt.reindex(rng)
    # irr.data_sim = pd.merge(irr.data_sim, df_0, how='right')
    # alt = pd.merge(alt, df_0, how='right')
          
    # pas sur que faire le ratio avec le simuler clear sky soit 
    # la meilleur forme surtout in the morning and evening transitioning
    # ratio = irr.ratio()
    #  
    #    
    # # ratio[ratio.between_time('16:00','08:00').index] = 100
    #     
    # glob_irr = irr.data_sim.multiply(ratio, axis=0)
    #    
    # glob_irr = glob_irr.reindex(pd.date_range(glob_irr.index[0], glob_irr.index[-1],freq='1H'))
    # glob_irr = glob_irr.replace([np.inf, -np.inf], np.nan)
    # glob_irr  = glob_irr.fillna(0)
    # alt = alt.reindex(pd.date_range(alt.index[0], alt.index[-1],freq='1H'), method='pad')
    # alt  = alt.fillna(0)
       
            
    # #===============================================================================
    # # Variables from Meteorological station (Only LCB)
    # #===============================================================================
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    # Files=glob.glob(InPath+"*")
    AttSta.setInPaths(InPath)
      
       
    station_names =AttSta.stations(['Head'])
    #[station_names.remove(k) for k in ['C06', 'C11', 'C12', 'C09', 'C16', 'C08', 'C17', 'C10']]
#     [station_names.remove(k) for k in ['C06','C11', 'C12','C09']]
    # [station_names.remove(k) for k in [ 'C17', 'C05','C06','C16']]
    # station_names.append('C10')
    # station_names =station_names + AttSta.stations(['Head', 'valley'])
    Files =AttSta.getatt(station_names,'InPath')
    net_LCB=LCB_net()
    net_LCB.AddFilesSta(Files)
#     net_LCB.dropstanan(perc=20, From=From, To = To)
     
    X_LCB = net_LCB.getvarallsta(var=var, by='H')
    # 
    # #------------------------------------------------------------------------------ 
    # 
    # # X_LCB =X_LCB- X_LCB.mean()
    # 
    # X_LCB= X_LCB["2015-04-01 00:00:00":"2015-10-01 00:00:00"]
    # X_LCB = X_LCB.groupby(lambda t: (t.hour)).mean().plot()
    # 
    # X_LCB = X_LCB.between_time('06:00','18:00')
    # 
    # 
    # a = X_LCB[['C10','C11','C12','C13', 'C14', 'C15']]
    # a.columns=['S7','S8','S9', 'S10', 'S11', 'S12']
    # 
    # a.plot(kind='scatter', x='S9', y='S10')
    # a.plot(kind='scatter', x='S9', y='S11')
    # 
    # plt.scatter(X_LCB['C14'],X_LCB['C12'])
    # 
    # 
    # 
    # a = a.between_time('05:00','06:00')
    # a.plot()
    # 
    # X_LCB['DiffC15C13']= X_LCB['C13']-X_LCB['C07']
    # X_LCB['DiffC15C12']= X_LCB['C12']-X_LCB['C07']
    # 
    # 
    # X_LCB[['DiffC15C13','DiffC15C12']].plot()
    # 
    # 
    # X_LCB['DIFF']=X_LCB['DiffC15C13'] - X_LCB['DiffC15C12']
    # 
    # 
    # 
    # 
    # X_LCB['DIFF'].plot() 
    # 
    # X_LCB['DiffC07C06']= X_LCB['C07']-X_LCB['C06']
    # X_LCB['DiffC07C05']= X_LCB['C07']-X_LCB['C05']
    # X_LCB['DIFF2']=X_LCB['DiffC07C06'] - X_LCB['DiffC07C05']
    # 
    # X_LCB['DIFF2'].plot() 
    # 
    # 
    # 
    # X_LCB[['DiffC15C13','DiffC15C12' ]].plot()
    # 
    # X_LCB[['C08','C12', 'C15', 'C13','C07']].between_time('12:00','12:00').plot()
    # 
    # #------------------------------------------------------------------------------ 
      
#     X = X_LCB.between_time('12:00','12:00')
#     X= X["2014-10-01 00:00:00":"2016-01-01 00:00:00"]
#     X = X.dropna(axis=0)
    # 
    # from scipy.stats.stats import pearsonr   
    # 
    # #===============================================================================
    # # Variables from meteorological stations
    # #===============================================================================
       
       
    net_sinda=LCB_net()
    net_inmet=LCB_net()
    net_iac=LCB_net()
    # net_lcb=LCB_net()
        
        
      
    # AttSta.setInPaths(InPath)
    
       
    Path_Sinda = '/home/thomas/PhD/obs-lcb/staClim/Sinda/obs_clean/Sinda/'
    Path_INMET ='/home/thomas/PhD/obs-lcb/staClim/INMET-Master/obs_clean/INMET/'
    Path_IAC ='/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/obs_clean/IAC/'
     
    
    AttSta_IAC = att_sta()
    AttSta_Inmet = att_sta()
    AttSta_Sinda = att_sta()
    
    
    AttSta_IAC.setInPaths(Path_IAC)
    AttSta_Inmet.setInPaths(Path_INMET)
    AttSta_Sinda.setInPaths(Path_Sinda)
      
    
    # #     station_names =AttSta.stations(values=['IAC'], region=region)
    # #     print station_names
    #  
    # region = [-24,-22,-48, -46]
    # station_names =AttSta.stations(region=region)
    #   
       
    # File_Sinda = glob.glob(Path_Sinda+"*")
    # File_INMET = glob.glob(Path_INMET+"*")
#     Files_IAC = glob.glob(Path_IAC+"*")
       
       
#     # station_names =AttSta.stations(['Ribeirao'])
#     stanames_IAC =  AttSta.stations(values=['IAC'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt})
#     stanames_Inmet = AttSta.stations(values=['Innmet'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt} )
#     stanames_Sinda = AttSta.stations(values=['Sinda'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt} )
#     [stanames_IAC.remove(x) for x in ['sm77','pr141','sg140'] if x in stanames_IAC ]
#     [stanames_Inmet.remove(x) for x in ['A531','A530'] if x in stanames_Inmet ]
     
#     Files_IAC =AttSta_IAC.getatt(stanames_IAC,'InPath')
#     Files_Inmet =AttSta_Inmet.getatt(stanames_Inmet,'InPath')
#     Files_Sinda =AttSta_Sinda.getatt(stanames_Sinda,'InPath')
    
#     Files_IAC =AttSta_IAC.getatt(['va89','bj13','br14','pr141','pc58','br14','hp208','ma46'],'InPath')
#     Files_Inmet =AttSta_Inmet.getatt(['A509'],'InPath')
#     Files_Sinda =AttSta_Sinda.getatt(stanames_Sinda,'InPath')
#     
         
#     net_sinda.AddFilesSta(Files_Sinda, net='Sinda')
#     net_inmet.AddFilesSta(Files_Inmet, net='INMET')
#     net_iac.AddFilesSta(Files_IAC, net='IAC')
      

# 
#     net_iac.dropstanan(perc=20, From=From, To = To)
#     net_inmet.dropstanan(perc=20, From=From, To = To)
#     net_sinda.dropstanan(perc=20, From=From, To = To, by='3H')
#     

#     X_iac = net_iac.getvarallsta(var=var,by='H',From=From, To = To)
#     X_inmet = net_inmet.getvarallsta(var=var,by='H',From=From, To = To)
#     X_sinda = net_sinda.getvarallsta(var=var,by='H',From=From, To = To)
#     

    # # with complete data
    # X_iac = net_iac.getvarallsta(var=var,stanames=['va89', 'pr141', 'br14', 'at7', 'cp119', 'ia32', 'ma46', 'sj70',
    #                                                'dv136', 'sg140', 'ca125', 'sb69', 'ep22'
    #                                                , 'su81','cs3'],by='H')
    #  
    # X_sinda = net_sinda.getvarallsta(var=var, by='H', stanames=['32526','30976','30975'])
    # X_inmet = net_inmet.getvarallsta(var=var,stanames=[ 'A706', 'A530','A531'], by='H')
        
    # Tmin
#     X_iac = net_iac.getvarallsta(var=var,stanames=['va89','br14', 'ex23', 'pr141', 'bj13'],by='H')
    #      
    # X_sinda = net_sinda.getvarallsta(var=var, by='H', stanames=['32526'])
    # X_inmet = net_inmet.getvarallsta(var=var,stanames=[ 'A530'], by='H')
        
        
    #Tmax
#     X_iac = net_iac.getvarallsta(var=var,stanames=['va89', 'br14', 'at7', 'cp119', 'ia32', 'ma46', 'sj70',
#                                                     'dv136', 'sg140', 'ca125', 'sb69', 'ep22'
#                                                     , 'su81','cs3'],by='H')
    #  
    # X_sinda = net_sinda.getvarallsta(var=var, by='H', stanames=['30976','30975'])
    # X_inmet = net_inmet.getvarallsta(var=var,stanames=[ 'A706','A531'], by='H')
    
     
    # X_inmet = X_inmet.fillna(X_inmet.mean())
    # X = pd.concat([X_iac, X_inmet,X_sinda,X_LCB], axis=1)
         
    #########################
    #X = X_inmet
    X = pd.concat([X_LCB], axis=1)
#     X= X["2015-03-01 00:00:00":"2016-01-01 00:00:00"]
    X = X.between_time('05:00','05:00')
    X.plot(subplots=True)
    X = X.dropna(axis=0,how='any')
#     # ##########
    plt.show()
#               
#     # #===============================================================================
#     # # Topographic parameters
#     # #===============================================================================
    params = pd.read_csv('/home/thomas/PhD/supmod/PCA_data/params_topo.csv', index_col=0)
#     params = params.dropna(axis=0)
#          
#     X_selected = pd.DataFrame([],index = X.index)
#          
#     # This should be implemented in the LCB_net class
#     for staname in params.index.values:
#         try:
#             X_selected = pd.concat([X_selected,X[staname]], axis=1)
#         except KeyError:
#             pass
#     X = X_selected
#          
#     X = X.between_time('03:00','03:00')
#     X= X["2014-11-01 00:00:00":"2016-01-01 00:00:00"]
#     #Fill na
#     X = X.dropna(axis=1,how='any')
#     X = X.fillna(X.mean()) # need to improve the prenchimento dos dados

        
#         
#         
#     #===============================================================================
#     # Perform the PCA Using Sklearn
#     #===============================================================================
    
    #------------------------------------------------------------------------------ 
    #Standardizing
    # from sklearn.preprocessing import StandardScaler
    #X_std = StandardScaler().fit_transform(X)
    
    # Another way to standardize using python
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
#     X_std =X # no scaling
    # X_std = X
         
    #X_std = X_std.dropna(axis=1,how='any')
    # To check if the variance is equal to zero
    np.var(X_std)
         
    #Create a PCA model with two principal components
    pca = PCA(3)
    # pca = PCA(1)
    pca.fit(X_std)
    #Get the components from transforming the original data.
    scores = pca.transform(X_std) #  or PCs
          
    # test
    # PC1=scores.T[0]
    # PC2=scores.T[1] + PC1
    # PC3= scores.T[2] +PC2
    #
         
    eigenvalues = pca.explained_variance_
    eigenvector = pca.components_ # or loading 
    # Make a list of (eigenvalue, eigenvector) tuples
    print eigenvector
    eigpairs = [(np.abs(eigenvalues[i]), eigenvector[i,:]) for i in range(len(eigenvalues))]
          
          
    # Reconstruct from the 2 dimensional scores 
    reconstruct_std = pca.inverse_transform(scores)
          
    #The residual is the amount not explained by the first two components
    residual_std=X_std-reconstruct_std
    reconstruct =reconstruct_std* X.std(axis=0).values + X.mean(axis=0).values
    df_stations = pd.DataFrame(data=reconstruct, index = X.index, columns = X.columns)
          
    residual=X-reconstruct
          
    reconstruct = pd.DataFrame(reconstruct, index=X.index, columns=X.columns)
#         
#     #===============================================================================
#     # reconstruct
#     #===============================================================================
#         
#     a = pd.concat([X, reconstruct], axis=1)
#         
#     plt.close('all')
#     a[['C08','C10']].plot(color=['r','b'],style=['-','-','--','--'])
#         
#     #===============================================================================
#     # Gfs variables 
#     #===============================================================================
#         
    inpath = "/home/thomas/PhD/supmod/PCA_data/"        
    df_gfs = pd.read_csv(inpath+'gfs_serie.csv', index_col =0, parse_dates=True )
    df_gfs.index =df_gfs.index + pd.offsets.Hour(12)# Problem I am using forecast 12h 
    df_gfs['TMP_950mb'] = df_gfs['TMP_950mb'] - 273.15
    print df_gfs
#            
#     cols_var = df_gfs.iloc[:,6:].columns
#            
#     # Data = pd.DataFrame(df_gfs.index)
#            
#     for c_v in cols_var:
#         newdf = pd.DataFrame(df_gfs[c_v] )
#         for staname in stanames: 
#             newdf[staname] = df_gfs[c_v]
#         del newdf[c_v]
#         if c_v == cols_var[0]:
#             data_gfs = pd.DataFrame(newdf.stack(), columns = [c_v])
#             print data_gfs
#         else:
#             data_gfs[c_v] = newdf.stack()
 
#         
#     #===============================================================================
#     # Test
#     #===============================================================================
    PC1=scores.T[0]
    PC2=scores.T[1]
    PC3= scores.T[2]
# #         
# #         
    res = pd.merge(X, df_gfs, left_index=True,right_index=True, how='left' )
 
    plt.scatter(PC1,res['TMP_950mb'],c='b')
    plt.scatter(PC2,res['TMP_950mb'],c='r')
    plt.scatter(PC3,res['TMP_950mb'],c='g')
    
    plt.show()
#     
#     
#     plt.show()
#     print res
#     plt.scatter(PC1, X['C09'],c='b')
#     plt.scatter(PC2, X['C09'],c='r')
#     plt.scatter(PC3, X['C09'],c='g')
#     plt.show()
# #         
#     plt.scatter(PC1, res['VVEL_950mb'],c='b')
#     plt.scatter(PC2, res['VVEL_950mb'],c='r')
#     plt.scatter(PC3, res['VVEL_950mb'],c='g')
#     plt.show()
# #         
# #         
#     plt.scatter(PC1, res['HGT_950mb'],c='b')
#     plt.scatter(PC2, res['HGT_950mb'],c='r')
#     plt.scatter(PC3, res['HGT_950mb'],c='g')
#     plt.show()
# #         
#     plt.scatter(PC1, res['RH_950mb'],c='b')
#     plt.scatter(PC2, res['RH_950mb'],c='r')
#     plt.scatter(PC3, res['RH_950mb'],c='g')
#     plt.show()
#
    
    d = {'PC1':PC1, 'PC2':PC2, 'PC3':PC3 }
    Pcs = pd.DataFrame(d, index = X.index)
    Pcs.plot()
    plt.grid()
    plt.show()
#         
#     #===============================================================================
#     # Correlation loadings and irradiation LCB
#     #===============================================================================
#         
#     I_15h= irr.data_sim.between_time('15:00','15:00').mean()
#         
#     I = []
#     for i in X.columns:
#         print i
#         I.append(I_15h[i])
#         
#     plt.scatter(eigenvector[0], I, c='b')
#     plt.scatter(eigenvector[1], I, c='r')
#     plt.scatter(eigenvector[2], I, c='g')
#         
#         
#     #===============================================================================
#     # Correlation between PC loadings and indexs28
#     #===============================================================================
#     plt.close('all')
#          
#          
#     for PC in [0,1,2]: 
#         print PC
#         for p in params.keys():
#             s_p =[]
#             for sta in X.keys():
#                 try:
#                     s_p.append(params[p][sta])
#                 except:
#                     print sta
#                     print 'passe pas'
#                     pass
#             fig, ax = plt.subplots()
#             plt.title(p)
#             plt.scatter(eigenvector[PC],s_p)
#             for i, txt in enumerate(X.columns):
#                 ax.annotate(txt, (eigenvector[PC][i],s_p[i]))
#             plt.show()
# #         
#         
#     #===============================================================================
#     # Correlation between PC loadings and real indexs
#     #===============================================================================
    plt.close('all')
    for PC in [0,1,2]:
        AttSta = att_sta()
        elev_real = AttSta.getatt(X.keys(),'Alt') 
        fig, ax = plt.subplots()
        plt.scatter(eigenvector[PC],elev_real)
        for i, txt in enumerate(X.columns):
            ax.annotate(txt, (eigenvector[PC][i],elev_real[i]))
        plt.show()
    #         
#         

#         
#     #===============================================================================
#     # Merge dataframes
#     #===============================================================================
#             
#     alt_stacked = alt.stack().reset_index()
#     alt_stacked.columns = ['Date', 'stations', 'Alt']
#     glob_irr_stacked = glob_irr.stack().reset_index()
#     glob_irr_stacked.columns = ['Date', 'stations', 'irr']
#     df_station_stacked = df_stations.stack().reset_index()
#     df_station_stacked.columns = ['Date', 'stations', var]
#     data_gfs = data_gfs.reset_index()
#     print cols_var
#     print data_gfs.columns
#     print data_gfs
#     data_gfs.columns = ['Date', 'stations']+list(cols_var)
#         
#         
#         
#         
#     # result = pd.merge(alt_stacked,glob_irr_stacked,on=['Date','stations'], how='inner')
#     # 
#     #     
#     # result = pd.merge(alt_stacked,glob_irr_stacked,on=['Date','stations'], how='inner')
#     # result = pd.merge(result,df_station_stacked,on=['Date','stations'], how='inner')
#     # result = pd.merge(result,data_gfs,on=['Date','stations'], how='inner')
#     # print result
#         
#         
#     result = pd.merge(df_station_stacked,p[0],on=['Date','stations'])
#     result = pd.merge(result,p[1],on=['Date','stations'], how='inner')
#     for i in p:
#         result = pd.merge(result,i,on=['Date','stations'], how='inner')
#         print result
#         
#     result.to_csv('/home/thomas/df_PCA_2.csv')
#           
#       
#  









