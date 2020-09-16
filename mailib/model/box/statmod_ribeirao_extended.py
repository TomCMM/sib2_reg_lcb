
from statmod_lib import *
from clima_lib.LCBnet_lib import *
# from mapstations import plot_local_stations
from numpy.testing.utils import measure
import datetime
import pickle
import sys
import matplotlib
import os
from toolbox import Alt_side, side_PC4


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)
 
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['axes.labelsize'] = 8
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.markersize'] = 8
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['legend.fontsize'] = 8
# plt.rc('legend',fontsize=20)


if __name__=='__main__':
    #===========================================================================
    # User input pathe
    #===========================================================================
    path_gfs = "/home/thomas/phd/local_downscaling/data/gfs_data/"
    path_indexes = "/home/thomas/phd/local_downscaling/data/indexes/"
    stamod_out_path = "/home/thomas/phd/dynmod/data/sim_140214/local_downscaling/"
    
    plt.style.use('ggplot')
    cwd = os.getcwd()
    print cwd
#     AttSta = att_sta('/home/thomas/phd/local_downscaling/data/topo_indexes/df_topoindex/df_topoindex.csv')
#     AttSta.attributes.dropna(how='all',axis=1, inplace=True)
#     AttSta.attributes.dropna(how='any',axis=0, inplace=True)
#     print AttSta.attributes

    From = "2015-03-01 00:00:00"
    To = "2016-01-01 00:00:00"

#===============================================================================
# Get scores predictors
#===============================================================================

    gfs_file = "ribeirao/df_features_scores.csv"  
#     gfs_file = "fnl_20140214.csv"
#     gfs_file = 'gfs_data_levels_analysis.csv'
#     gfs_file = "fnl_2015_basicvar.csv"
    df_gfs = pd.read_csv(path_gfs+gfs_file, index_col =0, parse_dates=True ) # GFS data
#     del  df_gfs['dirname']
#     del  df_gfs['basename']
#     del  df_gfs['time']
#     del  df_gfs['model']
#     del  df_gfs['InPath']
    df_gfs = df_gfs.dropna(axis=1,how='any') 
    df_gfs = df_gfs.dropna(axis=0,how='any')


#===============================================================================
# Create surface observations dataframe
#===============================================================================

    net_LCB = LCB_net()
    Path_LCB='/home/thomas/phd/obs/lcbdata/obs/full_sortcorr/'

    AttSta = att_sta('/home/thomas/phd/local_downscaling/data/metadata/df_topoindex_ribeirao.csv')
    AttSta.attributes['Alt'] = AttSta.attributes['Alt'].astype(float) # if not give problem in the stepwise libnear regression 
    AttSta.attributes.dropna(how='all',axis=1, inplace=True)
    AttSta.attributes.dropna(how='any',axis=0, inplace=True)

    AttSta.setInPaths(Path_LCB)
    stanames_LCB = AttSta.get_sta_id_in_metadata(values = ['Ribeirao'])
#     [stanames_LCB.remove(x) for x in ['C11','C12'] if x in stanames_LCB ] # Remove stations
    [stanames_LCB.remove(x) for x in ['C09', 'C13'] if x in stanames_LCB ] # Remove stations
    Files_LCB =AttSta.getatt(stanames_LCB,'InPath')
    net_LCB.AddFilesSta(Files_LCB)
    
    
    df_T = net_LCB.getvarallsta(var='Ta C', by='H', From=From, To = To )
    df_T = df_T.dropna(axis=0,how='any')
    
    df_Q = net_LCB.getvarallsta(var='Ua g/kg', by='H', From=From, To = To )
    df_Q = df_Q.dropna(axis=0,how='any')
    
    df_U = net_LCB.getvarallsta(var='U m/s', by='H', From=From, To = To )
    df_U = df_U.dropna(axis=0,how='any')
    
    df_V = net_LCB.getvarallsta(var='V m/s', by='H', From=From, To = To )
    df_V = df_V.dropna(axis=0,how='any')
    

    d = pd.concat([sdf_T, dfU, df_Vt], join='inner', axis=1))
    
    AttSta.addatt(df = side_PC4(stanames_LCB, AttSta))
    alt_side = pd.Series(Alt_side(AttSta.get_sta_id_in_metadata(values=["Ribeirao"]), AttSta), index =stanames_LCB, name="Alt_side")
    AttSta.addatt(df = alt_side)
     


#===============================================================================
# Create train and verify dataframe
#==============================================================================')
    df_gfs = df # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#------------------------------------------------------------------------------ 
# Select same index
#------------------------------------------------------------------------------
    df_attributes = AttSta.attributes[AttSta.attributes.index.isin(df.columns)]
    df = df.loc[:,df.columns.isin(df_attributes.index)]
    df = df[df.index.isin(df_gfs.index)]
    df_gfs = df_gfs[df_gfs.index.isin(df.index)]
    
#------------------------------------------------------------------------------ 
# train dataset
#------------------------------------------------------------------------------ 
#     df_train = df[:-len(df)/7]
    df_train = df
    df_gfs_train = df_gfs[df_gfs.index.isin(df_train.index)]
    
#------------------------------------------------------------------------------ 
# verify dataset
#------------------------------------------------------------------------------ 
#     df_verif = df[-len(df)/7:]
    df_verif=df
    df_gfs_verif = df_gfs[df_gfs.index.isin(df_verif.index)]
    df_verif = df_verif[df_verif.index.isin(df_gfs_verif.index)]
      
#=========================================
# Create model
#=========================================
#------------------------------------------------------------------------------ 
#    PCA
#------------------------------------------------------------------------------
    nb_pc = 6
    stamod = StaMod(df, AttSta)
    stamod.pca_transform(nb_PC=nb_pc, cov=True, standard=False, remove_mean1 =True, remove_mean0 =True)
    daily_scores = stamod.scores.groupby(lambda t: (t.hour)).mean(    daily_scores.plot()
#     plt.show()
#     stamod.plot_scores_ts()
#     stamod.plot_exp_var()
)

#------------------------------------------------------------------------------ 
#    Fit Pcs loadings
#------------------------------------------------------------------------------ 
#     df_models_loadings = stamod.stepwise_model(stamod.eigenvectors.T, df_attributes,lim_nb_predictors=4, constant=True, log=True)

    att_t = AttSta.attributes['Alt'].copy()
#     att_q = AttSta.attributes['Alt'].copy()
    att_u = AttSta.attributes['Alt'].copy()
    att_v = AttSta.attributes['Alt'].copy()
    
    att_t.index = [i + str('_T')for i in AttSta.attributes['Alt'].index]
#     att_q.index = [i + str('_Q')for i in AttSta.attributes['Alt'].index]
    att_u.index = [i + str('_U')for i in AttSta.attributes['Alt'].index]
    att_v.index = [i + str('_V')for i in AttSta.attributes['Alt'].index]
    
    att = pd.concat([att_t, att_u, att_v])

    predictors_loadings = [att,att,att,att,att,att]]

   aat = pd.concat([AttSta.attributes['Alt']]34)    aa.index = att.indexx
    predictors_loadings = aataataataataataat]
#                            pd.concat([AttSta.attributes['Alt']]*3),
#                            pd.concat([AttSta.attributes['Alt']]*3),
#                             pd.concat([AttSta.attributes['Alt']]*3)]
#      
#      
#     
    
    loadings = pd.DataFrame(stamod.eigenvectors.T)
    df_models_loadings =  stamod.fit_curvfit(predictors_loadings,
                                              loadings, fits=[lin,lin,lin,lin,lin,lin], plot=True)
#     df_models_loadings =  stamod.fit_curvfit(predictors_loadings,
#                                               loadings, fits=[multi_pol3_lin], plot=False)
# #------------------------------------------------------------------------------ 
# # #    Fit PCs scores
# # #------------------------------------------------------------------------------ 
    df_models_scores = stamod.stepwise_model(stamod.scores, df_gfs,lim_nb_predictors=7, constant=True, log=True)
# 
# # #===============================================================================
# # # Prediction
# # #===============================================================================
    res = stamod.predict_model(df_attributes, df_gfs_verif,df_models_loadings,
                                df_models_scores, model_loading_curvfit=True, observed_scores = False)
 
#===============================================================================
# model verification temperature directly from the stepwise regression
#===============================================================================
# 
    MAE =  stamod.skill_model(df_verif,res , metrics = metrics.mean_absolute_error, summary=True, plot_summary=True)
    MSE=  stamod.skill_model(df_verif,res , metrics = metrics.mean_squared_error, summary=True, plot_summary=True)
#      
    print "X"* 20
    print "Results"
    print "Spatially averaged MSE: " + str (MSE.mean(axis=1))
    print "Spatially averaged MAE: " + str (MAE.mean(axis=1))
    print "X"* 20
     
    print "X"* 20
    print "Results"
    print "Total averaged MSE: " + str (MSE.mean().mean())
    print "Total averaged MAE: " + str (MAE.mean().mean())
    print "X"* 20

#===============================================================================
# GFS temperature Error
#===============================================================================
    df_rec = pd.DataFrame(index= df_verif.index)
    for sta in df_verif.columns:
        df_rec = pd.concat([df_rec, df_gfs_verif["TMP_2maboveground"]], axis=1, join="outer") 
    df_rec.columns = df_verif.columns
    df_rec = df_rec-273.15
# #     error = df_verif - df_gfs_verif
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    
    MAE = mean_absolute_error(df_verif, df_rec)
    MSE = mean_squared_error(df_verif, df_rec)
    
#    
#     print MAE
#     print MSE
        
    print "X"* 20
    print "Results"
    print "large scale model temperture 2m spatially averaged MSE: " + str (MSE)
    print "large scale model temperture 2m Spatially averaged MAE: " + str (MAE)
    print "X"* 20
