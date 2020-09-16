#===============================================================================
# DESCRIPTION
#    Downscale SWAT input climatic variable from the 
#    Global Forecasting System in the Ribeirao Das Posses
#===============================================================================

from statmod_lib import *
from LCBnet_lib import *


if __name__=='__main__':

    #===============================================================================
    # Set up
    #===============================================================================
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full_sortcorr/' # Path data
    AttSta = att_sta()
    AttSta.setInPaths(InPath)
    From = "2015-03-01 00:00:00"
    To = "2016-01-01 00:00:00"
    
    params = pd.read_csv('/home/thomas/params_topo.csv', index_col=0) # topographic parameters
    print params.columns
    
    path_gfs = "/home/thomas/"        
    df_gfs = pd.read_csv(path_gfs+'gfs_data.csv', index_col =0, parse_dates=True ) # GFS data
    df_gfs = df_gfs[From:To]
    
    #===========================================================================
    # Rainfall
    #===========================================================================
    var = "Rc mm"
    station_names =AttSta.stations(['Ribeirao'])
#     [station_names.remove(k) for k in ['C05']] # Rc mm
#     [station_names.remove(k) for k in ['C05','C17']] # Rc mm
          
    Files =AttSta.getatt(station_names,'InPath')
    net_LCB=LCB_net()
    net_LCB.AddFilesSta(Files)
    X_LCB = net_LCB.getvarallsta(var=var, by='H', how='sum', From=From, To=To)
 
             
    X = X_LCB.dropna(axis=0,how='any')
#     df_verif = X[:50]
#     df_train = X[50:]
    df_train= X
    stamod = StaMod(df_train, AttSta)
    stamod.pca_transform(nb_PC=6, standard=False)
                
    stamod.plot_scores_ts()
#     stamod.plot_exp_var()
              
    # FIT LOADINGS
    params_loadings =  stamod.fit_loadings(params=["Alt","Alt","Alt","Alt","Alt","Alt"], curvfit=[pol,pol,pol,pol,pol,pol])
    stamod.plot_loading(params = params_loadings[0], params_topo= ["Alt","Alt","Alt","Alt","Alt","Alt"], curvfit=[pol,pol,pol,pol,pol,pol])
