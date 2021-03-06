#===============================================================================
# DESCRIPTION
#    Downscale SWAT input climatic variable from the 
#    Global Forecasting System in the Ribeirao Das Posses
# Author
#    Thomas Martin
#        PhD atmospheric sciences
#===============================================================================

from statmod_lib import *
from clima_lib.LCBnet_lib import *
plt.style.use('ggplot')
from datetime import timedelta
from images2gif import writeGif
from PIL import Image
import os





if __name__=='__main__':

    #===============================================================================
    # Set up
    #===============================================================================
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full_sortcorr/' # Path data
    Path_att = '/home/thomas/phd/local_downscaling/data/topo_indexes/df_topoindex/df_topoindex.csv' # Path attribut
    AttSta = att_sta(Path_att=Path_att)
    AttSta = att_sta()
    
    From = "2015-03-01 00:00:00"
    To = "2015-11-01 00:00:00"
    
    params = pd.read_csv(Path_att, index_col=0) # topographic parameters
    print params
    AttSta.setInPaths(InPath)
    AttSta.addatt(params)
    AttSta.setInPaths(InPath)
    AttSta.showatt()
    stations = AttSta.get_sta_id_in_metadata(values=["Ribeirao"])
    
    AttSta.addatt(df = side_PC4(stations))
    alt_side = pd.Series(Alt_side(AttSta.get_sta_id_in_metadata(values=["Ribeirao"])), index =stations, name="Alt_side")
    AttSta.addatt(df = alt_side)


#     print AttSta.showatt()
#     AttSta.setInPaths(InPath)
#     print AttSta.stations(values=["Alt_side"])
#     pd.Series(index=[])
     
#===============================================================================
# GFS predictor variables
#===============================================================================
     
    path_gfs = "/home/thomas/phd/local_downscaling/data/gfs_data/"
    df_gfs = pd.read_csv(path_gfs+'gfs_data_levels_analysis.csv', index_col =0, parse_dates=True ) # GFS data
#     df_gfs = df_gfs[From:To]
#     df_gfs.drop(['dirname','basename','time'],axis=1, inplace=True)
    del  df_gfs['dirname']
    del  df_gfs['basename']
    del  df_gfs['time']
    del  df_gfs['model']
    del  df_gfs['InPath']
    df_gfs = df_gfs.dropna(axis=1,how='all') 
    df_gfs = df_gfs.dropna(axis=0,how='all')
    print df_gfs.tail
    df_gfs = df_gfs["2015-03-01 03:00:00":"2015-11-01 03:00:00"]
    
    df_gfs.loc[:,['TMP_900mb','TMP_950mb','TMP_80maboveground','TMP_2maboveground']] =df_gfs.loc[:,['TMP_900mb','TMP_950mb','TMP_80maboveground','TMP_2maboveground']] - 273.15
#     df_gfs.loc[:,['HGT_900mb','HGT_950mb','PRES_80maboveground','PRES_surface']] = df_gfs.loc[:,['HGT_900mb','HGT_950mb','PRES_80maboveground','PRES_surface']]*10**-2
    df_gfs['wind_900mb'] ,a = cart2pol(df_gfs.loc[:,'UGRD_900mb'],df_gfs.loc[:,'VGRD_900mb'])
    df_gfs['wind_950mb'],a = cart2pol(df_gfs.loc[:,'UGRD_950mb'],df_gfs.loc[:,'VGRD_950mb'])
    df_gfs['wind_80maboveground'],a = cart2pol(df_gfs.loc[:,'UGRD_80maboveground'],df_gfs.loc[:,'VGRD_80maboveground'])
    df_gfs['wind_10maboveground'],a = cart2pol(df_gfs.loc[:,'UGRD_10maboveground'],df_gfs.loc[:,'VGRD_10maboveground'])
    
    
    df_gfs['q_500mb'] = q(df_gfs['RH_500mb'], df_gfs['TMP_500mb'],500 )
    df_gfs['q_550mb'] = q(df_gfs['RH_550mb'], df_gfs['TMP_550mb'],550 )
    df_gfs['q_600mb'] = q(df_gfs['RH_600mb'], df_gfs['TMP_600mb'],600 )
    df_gfs['q_650mb'] = q(df_gfs['RH_650mb'], df_gfs['TMP_650mb'],650 )
    df_gfs['q_700mb'] = q(df_gfs['RH_700mb'], df_gfs['TMP_700mb'],700 )
    df_gfs['q_750mb'] = q(df_gfs['RH_750mb'], df_gfs['TMP_750mb'],750 )
    df_gfs['q_800mb'] = q(df_gfs['RH_800mb'], df_gfs['TMP_800mb'],800 )
    df_gfs['q_850mb'] = q(df_gfs['RH_850mb'] ,df_gfs['TMP_850mb'],850 )
    df_gfs['q_900mb'] = q(df_gfs['RH_900mb'], df_gfs['TMP_900mb'],900 )
    df_gfs['q_950mb'] = q(df_gfs['RH_950mb'], df_gfs['TMP_950mb'],950 )
#     print df_gfs['PRES_surface']
    df_gfs['q_2maboveground'] = q(df_gfs['RH_2maboveground'], df_gfs['TMP_2maboveground'],df_gfs['PRES_surface']*10**-2 )
    

#     df_gfs = df_gfs.loc[:,['TMP_900mb', 'TMP_2maboveground','PRES_surface','HGT_950mb', 'VGRD_900mb']]
# 
    df_gfs_500mb = df_gfs.loc[:,['HGT_500mb','RH_500mb','TMP_500mb','UGRD_500mb','VGRD_500mb']]
    df_gfs_550mb = df_gfs.loc[:,['HGT_550mb','RH_550mb','TMP_550mb','UGRD_550mb','VGRD_550mb']]
    df_gfs_600mb = df_gfs.loc[:,['HGT_600mb','RH_600mb','TMP_600mb','UGRD_600mb','VGRD_600mb']]
    df_gfs_650mb = df_gfs.loc[:,['HGT_650mb','RH_650mb','TMP_650mb','UGRD_650mb','VGRD_650mb']]
    df_gfs_700mb = df_gfs.loc[:,['HGT_700mb','RH_700mb','TMP_700mb','UGRD_700mb','VGRD_700mb']]
    df_gfs_750mb = df_gfs.loc[:,['HGT_750mb','RH_750mb','TMP_750mb','UGRD_750mb','VGRD_750mb']]
    df_gfs_800mb = df_gfs.loc[:,['HGT_800mb','RH_800mb','TMP_800mb','UGRD_800mb','VGRD_800mb']]
    df_gfs_850mb = df_gfs.loc[:,['HGT_850mb','RH_850mb','TMP_850mb','UGRD_850mb','VGRD_850mb']]
    df_gfs_900mb = df_gfs.loc[:,['HGT_900mb','RH_900mb','TMP_900mb','UGRD_900mb','VGRD_900mb']]
    df_gfs_950mb = df_gfs.loc[:,['HGT_950mb','RH_950mb','TMP_950mb','UGRD_950mb','VGRD_950mb']]
    df_gfs_80m = df_gfs.loc[:,['PRES_80maboveground','RH_80maboveground','TMP_80maboveground','UGRD_80maboveground','VGRD_80maboveground']]
    df_gfs_2m = df_gfs.loc[:,['PRES_surface','RH_2maboveground','TMP_2maboveground','UGRD_10maboveground','VGRD_10maboveground']]

 
#     df_T = df_gfs.loc[:,['TMP_900mb','TMP_950mb','TMP_80maboveground','TMP_2maboveground']]
#     df_HGT =df_gfs.loc[:,['HGT_900mb','HGT_950mb','PRES_80maboveground','PRES_surface']]
#     df_RH =df_gfs.loc[:,['RH_850mb','RH_900mb','RH_950mb','RH_2maboveground']]
#     df_q =df_gfs.loc[:,['q_850mb','q_900mb','q_950mb','q_2maboveground']]
#     df_VGRD =df_gfs.loc[:,['wind_900mb','wind_950mb','wind_80maboveground','wind_10maboveground']]
  
    df_T = df_gfs.loc[:,['TMP_500mb','TMP_550mb','TMP_600mb','TMP_650mb','TMP_700mb','TMP_750mb','TMP_800mb','TMP_850mb','TMP_900mb','TMP_950mb','TMP_80maboveground','TMP_2maboveground']]
    df_HGT =df_gfs.loc[:,['HGT_500mb','HGT_550mb','HGT_600mb','HGT_650mb','HGT_700mb','HGT_750mb','HGT_800mb','HGT_850mb','HGT_900mb','HGT_950mb','PRES_80maboveground','PRES_surface']]
    df_RH =df_gfs.loc[:,['RH_500mb','RH_550mb','RH_600mb','RH_650mb','RH_700mb','RH_750mb','RH_800mb','RH_850mb','RH_900mb','RH_950mb','RH_2maboveground','RH_2maboveground']]
    df_VGRD =df_gfs.loc[:,['VGRD_500mb','VGRD_550mb','VGRD_600mb','VGRD_650mb','VGRD_700mb','VGRD_750mb','VGRD_800mb','VGRD_850mb','VGRD_900mb','VGRD_950mb','VGRD_80maboveground','VGRD_10maboveground']]
    df_RH =df_gfs.loc[:,['RH_500mb','RH_550mb','RH_600mb','RH_650mb','RH_700mb','RH_750mb','RH_800mb','RH_850mb','RH_900mb','RH_950mb','RH_2maboveground','RH_2maboveground']]  
    df_q =df_gfs.loc[:,['q_500mb','q_550mb','q_600mb','q_650mb','q_700mb','q_750mb','q_800mb','q_850mb','q_900mb','q_950mb','q_2maboveground','q_2maboveground']]
     
    df_gfs= pd.concat([df_T, df_HGT, df_RH, df_VGRD], axis=1)
#     df_gfs = df_gfs_2m
    
    
# #===============================================================================
# # TEST PCA on GFS
# #===============================================================================
#     stamod = StaMod(df_gfs, AttSta)
#     stamod.pca_transform(nb_PC=4, standard=False)
#     scores = stamod.scores
#     loadings = stamod.eigenvectors
#     scores.columns=['TPC1','TPC2','TPC3','TPC4']
#     PCs = scores
# 
    for i,predictor in enumerate([df_T, df_RH, df_q, df_HGT,df_VGRD]):
        stamod = StaMod(predictor, AttSta)
        stamod.pca_transform(nb_PC=4, standard=False)
        scores = stamod.scores
        loadings = stamod.eigenvectors

#         scores.plot()
#         stamod.plot_exp_var()
#         plt.show()
# # # #           
# # #         alt= [5860,5114,4420,3772,3165,2592,2051,1537,1046,579,80,2]
#         alt= [1046,579,80,2]
#         plt.figure()
#         for j in range(4):
#             print j+1
#             plt.plot(loadings.iloc[j,:],alt, label = "PC"+str(j+1))
#             plt.legend()
#             plt.ylabel('mean altitude in meter')
#             plt.ylabel('PC scores')
#         plt.show()
#             
    #     scores.columns=['TPC1','TPC2','TPC3','TPC4','TPC5','TPC6','TPC7','TPC8','TPC9','TPC10']
#         print scores
        if i ==0:
            PCs = scores
        else:
            PCs = pd.concat([PCs, scores], axis=1)
  
    PCs.columns=['TPC1','TPC2','TPC3','TPC4',
                    'RHPC1','RHPC2','RHPC3','RHPC4',
                    'QPC1','QPC2','QPC3','QPC4',
                    'HGTPC1','HGTPC2','HGTPC3','HGTPC4',
                    'VGRDPC1','VGRDPC2','VGRDPC3','VGRDPC4']
     
#     PCs.columns=['TPC1','TPC2','TPC3','TPC4','TPC5','TPC6','TPC7','TPC8',
#                 'RHPC1','RHPC2','RHPC3','RHPC4','RHPC5','RHPC6','RHPC7','RHPC8',
#                 'HGTPC1','HGTPC2','HGTPC3','HGTPC4','HGTPC5','HGTPC6','HGTPC7','HGTPC8',
#                 'VGRDPC1','VGRDPC2','VGRDPC3','VGRDPC4','VGRDPC5','VGRDPC6','VGRDPC7','VGRDPC8']
#     df_gfs = PCs
    df_gfs=pd.concat([PCs, df_gfs], axis=1)

#     # add specific variables
#     df_gfs['HGT_950mb-900mb'] = df_gfs['HGT_900mb'] - df_gfs['HGT_950mb']
#     df_gfs['PRES_surface-80maboveground'] = df_gfs['PRES_surface'] - df_gfs['PRES_80maboveground']
#     df_gfs['TMP_2maboveground-900mb'] =  df_gfs['TMP_900mb'] - df_gfs['TMP_2maboveground']
#     df_gfs['TMP_2maboveground-950mb'] =  df_gfs['TMP_950mb'] - df_gfs['TMP_2maboveground']
#     df_gfs['TMP_2maboveground-80m'] = df_gfs['TMP_2maboveground'] - df_gfs['TMP_80maboveground']
#     df_gfs['TMP_900mb-950mb'] = df_gfs['TMP_900mb'] - df_gfs['TMP_950mb']
#     df_gfs['wind_900mb-950mb'] = df_gfs['wind_900mb'] - df_gfs['wind_950mb']
#     df_gfs['RH_900mb-2maboveground'] = df_gfs['RH_950mb'] - df_gfs['RH_2maboveground']
#     df_gfs['q_2maboveground-80m'] = df_gfs['q_2maboveground'] - df_gfs['q_850mb']
#     df_gfs['q_950-900m'] = df_gfs['q_950mb'] - df_gfs['q_900mb']
#      
# # #===========================================================================
# # # Temperature
# # #===========================================================================
    var = "Ta C"
    station_names =AttSta.get_sta_id_in_metadata(['Ribeirao'])
#     station_names=['C10','C04','C05','C06','C07','C08','C09','C11','C12','C13','C14','C15']
#     station_names=['C10','C04','C05','C07','C08',https://soundcloud.com/paulhm'C09','C13','C14','C15']
#     print station_names
#     [station_names.remove(k) for k in ['C11','C12','C06']] # Tmin  
    Files =AttSta.getatt(station_names,'InPath')
    net_LCB=LCB_net()
    net_LCB.AddFilesSta(Files)
    X_LCB = net_LCB.getvarallsta(var=var, by='H', how='mean', From=From, To=To)                   
    X = X_LCB.dropna(axis=0,how='any')
    plt.plot(X)
#     plt.plot(X['C10'],c ='k', linewidth=5, label='valley center')
#     plt.plot(X['C09'],'--',c ='k', linewidth=5, label='ridge' )
#     plt.legend()
#     X.plot()
#     plt.grid(True, color='0.5')
#     plt.xlabel('Time')
#     plt.ylabel('Observed temperature (C)')
#     plt.savefig('/home/thomas/data.pdf', transparent=True)


 
    X = X[X.index.isin(df_gfs.index)]
    df_gfs = df_gfs[df_gfs.index.isin(X.index)]
      
     
#     df_train = X[:-len(X)/7]
    df_train = X
    df_gfs_train = df_gfs[df_gfs.index.isin(df_train.index)]
   
    stamod = StaMod(df_train, AttSta)
    stamod.pca_transform(nb_PC=4, standard=False)
    print stamod.eigenpairs
    print stamod.eigenvectors
    
    fig, ax = plt.subplots()
    plt.scatter(stamod.eigenvectors.loc[1,:], stamod.eigenvectors.loc[2,:])
    for i, txt in enumerate(stamod.eigenvectors.columns):
                ax.annotate(txt, (stamod.eigenvectors.iloc[0,i], stamod.eigenvectors.iloc[1,i]))
    plt.show()

# #===============================================================================
# # 
# #===============================================================================
# #     
#     
#     df_verif = X[-len(X)/7:]
    df_verif=X
    df_gfs_verif = df_gfs[df_gfs.index.isin(df_verif.index)]
    df_verif = df_verif[df_verif.index.isin(df_gfs_verif.index)]
     
                      
#     stamod.plot_scores_ts(output='/home/thomas/PCts.pdf')
#     stamod.plot_exp_var(output='/home/thomas/expvar.pdf')
# #     plt.show()
#    
# #     # FIT LOADINGS
    params_loadings =  stamod.fit_loadings(params=["Alt","Alt","Alt","Alt"], fit=[pol3,pol3,piecewise_linear,lin])
    stamod.plot_loading(params_fit = params_loadings[0],params_topo=["Alt","Alt","Alt","Alt"], fit=[pol3,pol3,piecewise_linear,lin], output='/home/thomas/')
#     #fit scores
#    
    stamod.stepwise(df_gfs_train,lim_nb_predictors=4)
#     model = stamod.stepwise(df_gfs_train)
#        
# #===========================================================================
# # Specific humidity
# #===========================================================================
# #     var = "Ua g/kg"
# #     station_names =AttSta.stations(['Ribeirao'])
# # #     [station_names.remove(k) for k in ['C11', 'C09','C17','C18','C19','C16']] # Tmin
# # #     [station_names.remove(k) for k in ['C13','C11','C09']] # Tmin
# # #     [station_names.remove(k) for k in ['C13','C11','C12']] 
# # #     [station_names.remove(k) for k in ['C13','C16','C11','C12']]
# # #     [station_names.remove(k) for k in ['C17','C08']]
# #              
# #     Files =AttSta.getatt(station_names,'InPath')
# #     net_LCB=LCB_net()
# #     net_LCB.AddFilesSta(Files)
# #     X_LCB = net_LCB.getvarallsta(var=var, by='H',how='mean', From=From, To=To)
# # #     X_LCB = X_LCB.between_time('03:00','03:00')
# # #     X_LCB.plot()
# #     X = X_LCB.dropna(axis=0,how='any')
# #      
# #     df_train =X
# # #     df_train = X[:-len(X)/4]
# #     df_gfs_train = df_gfs[df_gfs.index.isin(df_train.index)]
# #       
# #     stamod = StaMod(df_train, AttSta)
# #     stamod.pca_transform(nb_PC=4, standard=False)
# #            
# # #     df_verif = X[-len(X)/4:]
# #     df_verif=X
# #     df_gfs_verif = df_gfs[df_gfs.index.isin(df_verif.index)]
# #     df_verif = df_verif[df_verif.index.isin(df_gfs_verif.index)]
# #       
# # #     stamod.plot_scores_ts()
# # #     stamod.plot_exp_var()
# # #     plt.show()
# #                 
# #     # FIT LOADINGS
# #     params_loadings =  stamod.fit_loadings(params=["Alt","Alt","Alt","Alt"], fit=[lin,pol2,piecewise_linear2,lin])
# # #     stamod.plot_loading(params_fit = params_loadings[0], params_topo= ["Alt","Alt","Alt","Alt"], fit=[lin,pol2,piecewise_linear2,lin])
# #                
#   
# #     model = stamod.stepwise(df_gfs_train,lim_nb_predictors=10)
# # #     model = stamod.stepwise(df_gfs_train)
# #===========================================================================
# # Wind speed
# #===========================================================================
# #     var = "Sm m/s"
# #     station_names =AttSta.stations(['Ribeirao'])
# #     [station_names.remove(k) for k in [ 'C09','C06','C13','C16']] # Sm m/s # 
# # #     [station_names.remove(k) for k in ['C06','C16']] # V m/s # 
# # #     [station_names.remove(k) for k in ['C13']] # U m/s # 
# # #     [station_names.remove(k) for k in ['C06']] # head SM m/s #     
# #         
# #     print station_names
# #     Files =AttSta.getatt(station_names,'InPath')
# #     net_LCB=LCB_net()
# #     net_LCB.AddFilesSta(Files)
# #     Path_att = '/home/thomas/params_topo.csv' # Path attribut
# #     AttSta = att_sta(Path_att=Path_att)
# #     AttSta.showatt()
# #         
# #     X_LCB = net_LCB.getvarallsta(var=var, by='3H', From=From, To=To)
# # #     X_LCB = X_LCB.between_time('03:00','03:00')
# #           
# # #     X_LCB.plot()
# #               
# #     X = X_LCB.dropna(axis=0,how='any')
# # #     df_train = X[:-len(X)/3]
# #     df_train=X
# #     df_gfs_train = df_gfs[df_gfs.index.isin(df_train.index)]
# #      
# #     stamod = StaMod(df_train, AttSta)
# #     stamod.pca_transform(nb_PC=3, standard=False)
# #      
# # #     df_verif = X[-len(X)/4:]
# #     df_verif=X
# #     df_gfs_verif = df_gfs[df_gfs.index.isin(df_verif.index)]
# #     df_verif = df_verif[df_verif.index.isin(df_gfs_verif.index)]  
# # #     stamod.plot_scores_ts()
# # #     stamod.plot_exp_var()
# # #     plt.show()
# # #     # FIT LOADINGS
# # #     params_loadings =  stamod.fit_loadings(params=["Alt","Alt",'Alt',"Alt"], fit=[lin,lin,lin,pol])
# # #     stamod.plot_loading(params_fit = params_loadings[0], params_topo= ["Alt","Alt",'Alt',"Alt"], fit=[lin,lin,lin,pol])
# #            
# #     params_loadings =  stamod.fit_loadings(params=["topex","topex_w",'Alt'], fit=[lin,pol3,lin])
# # #     stamod.plot_loading(params_fit = params_loadings[0], params_topo= ["topex","topex_w",'Alt'], fit=[lin,pol3,lin])
# # #         
# # #     #fit scores
# # # #     df_gfs_r = df_gfs.resample("3H").mean()
# # #     df_gfs_r = df_gfs
# # #        
# # #     U = df_gfs_r['UGRD_850mb']
# # #     V =  df_gfs_r['VGRD_850mb']
# # #             
# # #     mean_speed , mean_dir = cart2pol( U, V )
# # #     U_rot, V_rot = PolarToCartesian(mean_speed,mean_dir, rot=-45)
# # #        
# # #     params_scores = stamod.fit_scores([mean_speed, V_rot, df_gfs_r['UGRD_10maboveground']],fit=[lin,lin,lin])
# # #      
# # # #     [T,fr], params_fit = params_scores[0], fit=[lin,lin]
# # #      
# # #     stamod.plot_scores([mean_speed, V_rot, df_gfs_r['UGRD_10maboveground']], params_fit = params_scores[0],fit=[lin,lin,lin])
# # 
# #     model = stamod.stepwise(df_gfs_train,lim_nb_predictors=2)
# # #     model = stamod.stepwise(df_gfs_train)
#   
# #===============================================================================
# # model verification temperature directly from the stepwise rgression
# #===============================================================================
# 
# 
# 
#     res = stamod.predict_model(stamod.topo_index, df_gfs_verif)
#     ME =  stamod.skill_model(df_verif,res , metrics = mean_error, plot_summary=True, summary=True)
#     MAE =  stamod.skill_model(df_verif,res , metrics = metrics.mean_absolute_error, plot_summary=True, summary=True)
#     MSE=  stamod.skill_model(df_verif,res , metrics = metrics.mean_squared_error, plot_summary=True, summary=True)
#     corr = stamod.skill_model(df_verif,res , metrics = metrics.r2_score, plot_summary=True, summary=True)
#     exp_var = stamod.skill_model(df_verif,res , metrics = metrics.explained_variance_score, plot_summary=True, summary=True)
# 
#     print ME
#   
#     print "X"* 20
#     print "Results"
#     print "Total mean ME: " + str (ME.mean())
#     print "Total mean MSE: " + str (MSE.mean())
#     print "Total mean MAE: " + str (MAE.mean())
#     print "Total mean corr: " + str (corr.mean())
#     print "Total mean expvar: " + str (exp_var.mean())
#     print "X"* 20
#   
# #===============================================================================
# # model verification by hours
# #===============================================================================
# #     res = stamod.predict_model(stamod.topo_index, df_gfs_verif)
# #     data = res['predicted'].sum(axis=2)
# #     df_rec = pd.DataFrame(data, columns = df_verif.columns, index =df_verif.index) # should improve this
# # 
# #     plt.figure()
# #     plt.plot(X['C04'],'-',c='b', label='Observed')
# #     plt.plot(df_rec['C04'],'-',c='r', label='Calculated temperature in the valley (C04)')
# #     plt.ylabel('Temperature (C) in the valley (C04)')
# #     plt.xlabel('Time')
# #     plt.legend()
# #     plt.savefig('/home/thomas/verifc04.pdf', transparent=True)
# # 
# #     plt.figure()
# #     plt.plot(X['C07'],'-',c='b', label='Observed')
# #     plt.plot(df_rec['C07'],'-',c='r', label='Calculated')    
# #     plt.ylabel('Temperature (C) in the slope (C07)')
# #     plt.xlabel('Time')
# #     plt.legend()
# #     plt.savefig('/home/thomas/verifc07.pdf', transparent=True)
#     
# #      
# #      
# # ## gfs
# #     df_rec = pd.DataFrame(index= df_verif.index)
# #     for sta in df_verif.columns:
# #         df_rec = pd.concat([df_rec, df_gfs["TMP_2maboveground"]], axis=1, join="outer") 
# #     df_rec.columns = df_verif.columns
# # #      
# #     error = df_verif - df_rec
# #     from sklearn.metrics import mean_absolute_error, mean_squared_error
# #        
# #        
# #     print "X"* 20
# #     print "Results"
# #     print "Total mean MSE: " + str (mean_squared_error(df_verif, df_rec))
# #     print "Total mean MAE: " + str (mean_absolute_error(df_verif, df_rec))
# #     print "X"* 20
# #       
# #       
# #     hours = ['03:00', '09:00', '15:00', '21:00']
# #        
# #     def by_hours(hour):
# #         print "=" * 20
# #         print hour
# #         print "=" * 20
# #         print "MSE"
# #         print  mean_squared_error(df_verif.between_time(hour, hour), df_rec.between_time(hour,hour))
# #         print "MAE"
# #         print mean_absolute_error(df_verif.between_time(hour,hour), df_rec.between_time(hour,hour))
# #         plt.hist(error['C08'].between_time(hour,hour),30,facecolor="blue", alpha=0.75, label=str(hour)+ 'H')
# #         plt.legend()
# #         plt.show()
# #           
# #     for hour in hours:
# #         by_hours(hour)
# #
# # #===============================================================================
# # # model verification by wind speed at ridge and irradiation
# # #===============================================================================
# #     irr = LCB_Irr()
# #     inpath_obs = '/home/thomas/PhD/obs-lcb/LCBData/obs/Irradiance/data/'
# #     files_obs = glob.glob(inpath_obs+"*")
# #     irr.read_obs(files_obs)
# #     data = irr.data_obs
# #     data.to_csv('/home/thomas/irr.csv')
# #     inpath_sim='/home/thomas/Irradiance_rsun_lin2_lag0_glob_df.csv'
# #     irr.read_sim(inpath_sim)
# #     ratio = irr.ratio()
# #     ratio = ratio.resample('H').mean()
# # #     print ratio
# #     ratio = ratio[ratio.index.isin(df_verif.index)]
# #      
# # # 
# #     sta9_wind = net_LCB.getvarallsta("Sm m/s", ['C09'])
# #     sta9_wind = sta9_wind[sta9_wind.index.isin(df_verif.index)]
# # #     
# # #     sta9_rain = net_LCB.getvarallsta("Rc mm", ['C09'], by='3H')
# # #     sta9_rain = sta9_rain[sta9_rain.index.isin(df_verif.index)]
# # #     
# # #     
# # #     
# # #     print sta9_wind.index
# # #     print df_verif.index
# # #      
# # #     for score in stamod.scores:
# # #         print score
# # #         c=iter(['k','r','b','g'])
# # #         for hour in hours:
# # #             print hour
# # #             plt.scatter(error['C10'].between_time(hour, hour),stamod.scores[score].between_time(hour, hour), c=c.next(), s=50, alpha=0.6, label = str(hour) + "hour")
# # #              
# # #             plt.ylabel("Principal component " + str(score))
# # #             plt.xlabel('Error (C)')
# # #         plt.legend()
# # #         plt.show()
# # #     
# #  
# #     c=iter(['k','r','b','g'])
# #     for hour in hours:
# #         print hour
# #         plt.scatter(error.mean(axis=1).between_time(hour, hour),ratio.between_time(hour, hour), c=c.next(), s=50, alpha=0.6, label = str(hour) + "hour")
# #           
# #         plt.ylabel("Ratio Iobs/Icalc")
# #         plt.xlabel('Error (C)')
# #     plt.legend()
# #     plt.show()
# # #      
#  
#  
#  



    