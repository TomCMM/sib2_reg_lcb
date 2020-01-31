
"""
DESCRIPTION
    High resolution climate downscaling data pipeline for the Ribeirao Das Posses

"""

# import os # TODO WARNING TEST MODE
# os.environ["MKL_NUM_THREADS"] = "5"
# os.environ["NUMEXPR_NUM_THREADS"] = "5"
# os.environ["OMP_NUM_THREADS"] = "5"

from matplotlib.ticker import FormatStrFormatter
from sklearn import metrics
from scipy import stats
from scipy.stats import pearsonr
from functools import reduce
from keras.models import load_model, save_model
import xarray
import pandas as pd
from pandas.tseries import offsets

import joblib
import itertools
import os
import random as rm
from collections import defaultdict
import luigi
from pathlib import Path # amazing to deal with path
import matplotlib.pyplot as plt
import datetime

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mailib.stanet.lcbstanet import Att_Sta
from mailib.toolbox.fit_func import *
from mailib.model.statmod.statmod_lib import StaMod
from mailib.toolbox.tools import common_index
from mailib.model.nn.structures import DNN_ribeirao
from mailib.workflow.workflow import Framework

from maihr_config import Maihr_Conf

# Fix specific seed for reproductibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rm.seed(0)

plt.rcParams["figure.figsize"] = (7.2,14)

red = '#990000'
blue = '#00008B'

def get_encoded_hours(index):
    """


    :param index: pandas datetime index
    :return: dataframe, encoded months
    """

    hours = np.unique(index.hour)

    df = pd.DataFrame([index.hour == hour for hour in hours])
    df = pd.DataFrame(df.T.values, index=index, columns=hours).astype(int)
    df = df # TODO: Reducing intensity of the encoded vector
    return df

def print_skill_articleII(stamod,df_gcm_t,df_gcm_q,df_gcm_u,df_gcm_v):
    # # # # print linear correlation coefficient
    # print('PC1')
    # print('Temperature correlation PC1')
    # print_linear_correl_select(stamod['T']['df_models_scores']['lin'], stamod['T']['log'], pc_nb=1)
    # print('Specific humidity correlation')
    # print_linear_correl_select(stamod['Q']['df_models_scores']['lin'], stamod['Q']['log'], pc_nb=1)
    # print('Zonal wind')
    # print_linear_correl_select(stamod['U']['df_models_scores']['lin'], stamod['U']['log'], pc_nb=1)
    # print('Meridional wind')
    # print_linear_correl_select(stamod['V']['df_models_scores']['lin'], stamod['V']['log'], pc_nb=1)
    #
    # print('PC2')
    # # print linear correlation coefficient
    # print('Temperature correlation PC2')
    # print_linear_correl_select(stamod['T']['df_models_scores']['lin'], stamod['T']['log'], pc_nb=2)
    # print('Specific humidity correlation')
    # print_linear_correl_select(stamod['Q']['df_models_scores']['lin'], stamod['Q']['log'], pc_nb=2)
    # print('Zonal wind')
    # print_linear_correl_select(stamod['U']['df_models_scores']['lin'], stamod['U']['log'], pc_nb=2)
    # print('Meridional wind')
    # print_linear_correl_select(stamod['V']['df_models_scores']['lin'], stamod['V']['log'], pc_nb=2)
    #
    # print('PC3')
    # # print linear correlation coefficient
    # print_linear_correl_select(stamod['T']['df_models_scores']['lin'], stamod['T']['log'], pc_nb=3)
    # print('Specific humidity correlation')
    # # print_linear_correl_select(stamod['Q']['df_models_scores']['lin'], stamod['Q']['log'], pc_nb=3)
    # print('Zonal wind')
    # # print_linear_correl_select(stamod['U']['df_models_scores']['lin'], stamod['U']['log'], pc_nb=3)
    # print('Meridional wind')
    # # print_linear_correl_select(stamod['V']['df_models_scores']['lin'], stamod['V']['log'], pc_nb=3)
    #
    # print('PC4')
    # print_linear_correl_select(stamod['T']['df_models_scores']['lin'], stamod['T']['log'], pc_nb=4)
    # # print('Specific humidity correlation')
    # # print_linear_correl_select(stamod['Q']['df_models_scores']['lin'], stamod['Q']['log'], pc_nb=4)
    # # print('Zonal wind')
    # # print_linear_correl_select(stamod['U']['df_models_scores']['lin'], stamod['U']['log'], pc_nb=4)
    # # print('Meridional wind')
    # # print_linear_correl_select(stamod['V']['df_models_scores']['lin'], stamod['V']['log'], pc_nb=4)

    df_obs_T = stamod['T']['stamod'].df.loc[stamod['T']['res']['lin']['index'], :]
    df_obs_Q = stamod['Q']['stamod'].df.loc[stamod['Q']['res']['lin']['index'], :]
    df_obs_U = stamod['U']['stamod'].df.loc[stamod['U']['res']['lin']['index'], :]
    df_obs_V = stamod['V']['stamod'].df.loc[stamod['V']['res']['lin']['index'], :]

    df_gcm_t_m = pd.concat([df_gcm_t for d in range(len(df_obs_T.columns))],axis=1)
    df_gcm_t_m.columns = df_obs_T.columns
    df_gcm_q_m = pd.concat([df_gcm_q for d in range(len(df_obs_Q.columns))],axis=1)
    df_gcm_q_m.columns = df_obs_Q.columns
    df_gcm_u_m = pd.concat([df_gcm_u for d in range(len(df_obs_U.columns))],axis=1)
    df_gcm_u_m.columns = df_obs_U.columns
    df_gcm_v_m = pd.concat([df_gcm_v for d in range(len(df_obs_V.columns))],axis=1)
    df_gcm_v_m.columns = df_obs_V.columns

    print('RAW GCM MEAN ABSOLUTE VALUES T,Q,U,V')
    print(metrics.mean_absolute_error(df_obs_T,df_gcm_t_m.loc[df_obs_T.index,:]))
    print(metrics.mean_absolute_error(df_obs_Q,df_gcm_q_m.loc[df_obs_Q.index,:]))
    print(metrics.mean_absolute_error(df_obs_U,df_gcm_u_m.loc[df_obs_U.index,:]))
    print(metrics.mean_absolute_error(df_obs_V,df_gcm_v_m.loc[df_obs_V.index,:]))

    print('RAW GCM RMSE T,Q,U,V')

    print(np.power(metrics.mean_squared_error(df_obs_T,df_gcm_t_m.loc[df_obs_T.index,:]), 1. / 2))
    print(np.power(metrics.mean_squared_error(df_obs_Q,df_gcm_q_m.loc[df_obs_Q.index,:]), 1. / 2))
    print(np.power(metrics.mean_squared_error(df_obs_U,df_gcm_u_m.loc[df_obs_U.index,:]), 1. / 2))
    print(np.power(metrics.mean_squared_error(df_obs_V,df_gcm_v_m.loc[df_obs_V.index,:]), 1. / 2))




    print('Final lin model mean MAE, RMSE, pearson R')
    print('Temperature')
    print(stamod['T']['verif']['MAE']['lin'].mean().mean())
    print(stamod['T']['verif']['RMSE']['lin'].mean().mean())
    print(stamod['T']['verif']['R']['lin'].mean().mean())

    print('Specific humidity')
    print(stamod['Q']['verif']['MAE']['lin'].mean().mean())
    print(stamod['Q']['verif']['RMSE']['lin'].mean().mean())
    print(stamod['Q']['verif']['R']['lin'].mean().mean())

    print('Zonal wind')
    print(stamod['U']['verif']['MAE']['lin'].mean().mean())
    print(stamod['U']['verif']['RMSE']['lin'].mean().mean())
    print(stamod['U']['verif']['R']['lin'].mean().mean())

    print('Meridional wind')
    print(stamod['V']['verif']['MAE']['lin'].mean().mean())
    print(stamod['V']['verif']['RMSE']['lin'].mean().mean())
    print(stamod['V']['verif']['R']['lin'].mean().mean())

    print('o' * 10)
    print('Comparioson MAE models dnn, lin, lin-q')
    print('Temperature')
    print(stamod['T']['verif']['MAE']['dnn'].mean().mean())
    print(stamod['T']['verif']['MAE']['lin'].mean().mean())
    print(stamod['T']['verif']['MAE']['lin-q'].mean().mean())

    print('Comparioson RMSE models dnn, lin, lin-q  -> ON STATION C10')
    print('Temperature')
    print(stamod['T']['verif']['RMSE']['dnn'].mean().mean())
    print(stamod['T']['verif']['RMSE']['lin'].mean().mean())
    print(stamod['T']['verif']['RMSE']['lin-q'].mean().mean())

    print('Comparioson R models dnn, lin, lin-q')

    print(stamod['T']['verif']['R']['dnn'].mean().mean())
    print(stamod['T']['verif']['R']['lin'].mean().mean())
    print(stamod['T']['verif']['R']['lin-q'].mean().mean())

    print('Specific humidity')
    print('Comparioson MAE models dnn, lin, lin-q')

    print(stamod['Q']['verif']['MAE']['dnn'].mean().mean())
    print(stamod['Q']['verif']['MAE']['lin'].mean().mean())
    print(stamod['Q']['verif']['MAE']['lin-q'].mean().mean())
    print('Comparioson RMSE models dnn, lin, lin-q')

    print(stamod['Q']['verif']['RMSE']['dnn'].mean().mean())
    print(stamod['Q']['verif']['RMSE']['lin'].mean().mean())
    print(stamod['Q']['verif']['RMSE']['lin-q'].mean().mean())

    print('Comparioson R models dnn, lin, lin-q')

    print(stamod['Q']['verif']['R']['dnn'].mean().mean())
    print(stamod['Q']['verif']['R']['lin'].mean().mean())
    print(stamod['Q']['verif']['R']['lin-q'].mean().mean())

    print('Zonal Wind === station C07 C14')
    print('Comparioson MAE models dnn, lin, lin-q')

    print(stamod['U']['verif']['MAE']['dnn'].mean().mean())
    print(stamod['U']['verif']['MAE']['lin'].mean().mean())
    print(stamod['U']['verif']['MAE']['lin-q'].mean().mean())
    print('Comparioson RMSE models dnn, lin, lin-q')

    print(stamod['U']['verif']['RMSE']['dnn'].mean().mean())
    print(stamod['U']['verif']['RMSE']['lin'].mean().mean())
    print(stamod['U']['verif']['RMSE']['lin-q'].mean().mean())

    print('Comparioson R models dnn, lin, lin-q')

    print(stamod['U']['verif']['R']['dnn'].mean().mean())
    print(stamod['U']['verif']['R']['lin'].mean().mean())
    print(stamod['U']['verif']['R']['lin-q'].mean().mean())

    print('Meridional Wind === station C07 C14')
    print('Comparioson MAE models dnn, lin, lin-q')

    print(stamod['V']['verif']['MAE']['dnn'].mean().mean())
    print(stamod['V']['verif']['MAE']['lin'].mean().mean())
    print(stamod['V']['verif']['MAE']['lin-q'].mean().mean())
    print('Comparioson RMSE models dnn, lin, lin-q')

    print(stamod['V']['verif']['RMSE']['dnn'].mean().mean())
    print(stamod['V']['verif']['RMSE']['lin'].mean().mean())
    print(stamod['V']['verif']['RMSE']['lin-q'].mean().mean())

    print('Comparioson R models dnn, lin, lin-q')

    print(stamod['V']['verif']['R']['dnn'].mean().mean())
    print(stamod['V']['verif']['R']['lin'].mean().mean())
    print(stamod['V']['verif']['R']['lin-q'].mean().mean())

    print('done')

    print('OPPAAAAA')

def plot_lr_Uridge_irr(data, ax=None, marker=None, label=None):
    cmap = plt.cm.get_cmap("RdBu_r")

    sc = ax.scatter(data.loc[:, 'data'], data.loc[:, 'wind'], c=data.loc[:, 'irr'], cmap=cmap, s=100, marker=marker, label=label)

    # ax.set_ylim((0, 20))

    ax.set_ylabel(r'Wind speed normalized obs - gcm ', fontsize=14)
    ax.grid(True)
    ax.axvline(x=0, color='k', linewidth=5, alpha=0.5, zorder=0)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=5, alpha=0.5, zorder=0)


    return sc,ax

def perform_pca_byvar(df_gfs):
    gfs_index = df_gfs.index
    # df_gfs = pd.DataFrame(normalize(df_gfs), columns=df_gfs.columns, index=df_gfs.index)
    # df_gfs = pd.DataFrame(normalize(df_gfs,axis=0), columns=df_gfs.columns, index=df_gfs.index)

    col_z = df_gfs.filter(regex='z_var').columns
    col_t = df_gfs.filter(regex='t_var').columns
    col_q = df_gfs.filter(regex='q_var').columns
    col_w = df_gfs.filter(regex='w_var').columns
    col_r = df_gfs.filter(regex='r_var').columns
    col_cc = df_gfs.filter(regex='cc_var').columns
    col_u = df_gfs.filter(regex='u_var').columns
    col_v = df_gfs.filter(regex='v_var').columns

    col_filtered = list(itertools.chain.from_iterable([col_z, col_t, col_q, col_w, col_r, col_cc, col_u, col_v]))


    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=10))])

    df_gfs_rest = df_gfs.drop(col_filtered, axis=1)

    df_z = pd.DataFrame(pipeline.fit_transform(df_gfs.loc[:,col_z]), index=gfs_index)
    df_t = pd.DataFrame(pipeline.fit_transform(df_gfs.loc[:,col_t]), index=gfs_index)
    df_q = pd.DataFrame(pipeline.fit_transform(df_gfs.loc[:,col_q]), index=gfs_index)
    df_w = pd.DataFrame(pipeline.fit_transform(df_gfs.loc[:,col_w]), index=gfs_index)
    df_r = pd.DataFrame(pipeline.fit_transform(df_gfs.loc[:,col_r]), index=gfs_index)
    df_cc = pd.DataFrame(pipeline.fit_transform(df_gfs.loc[:,col_cc]), index=gfs_index)
    df_u = pd.DataFrame(pipeline.fit_transform(df_gfs.loc[:,col_u]), index=gfs_index)
    df_v = pd.DataFrame(pipeline.fit_transform(df_gfs.loc[:,col_v]), index=gfs_index)
    df_gfs_rest = pd.DataFrame(pipeline.fit_transform(df_gfs_rest), index=gfs_index)

    df_gfs_pca = pd.concat([df_z, df_t, df_q, df_w, df_r, df_cc, df_u, df_v, df_gfs_rest], axis=1, join='inner')

    df_gfs_pca = pd.concat([df_q, df_w, df_cc], axis=1, join='inner')

    df_gfs_pca.columns = range(len(df_gfs_pca.columns))

    # df_gfs_pca  = pd.DataFrame(scaler.fit_transform(df_gfs_pca.values), index= df_gfs_pca.index, columns=df_gfs_pca.columns)

    df_hours = get_encoded_hours(df_gfs.index)
    df_hours.columns = [ str(c) + 'hour' for c in df_hours.columns]

    # df_gfs_pca = pd.concat([df_gfs,df_gfs_pca,df_hours],axis=1)
    df_gfs_pca = pd.concat([df_gfs],axis=1)

    return df_gfs_pca

def print_linear_correl_select(df_models_scores, log, pc_nb=None):
    final_set = df_models_scores.loc[pc_nb,'predictor']
    simple_linear_correl_pc1 = log[pc_nb][0]

    print(simple_linear_correl_pc1[simple_linear_correl_pc1.loc[:,'candidate'].isin(final_set)])

def print_explained_var(model_T, model_Q, model_U, model_V):
    print('Explained variance')
    print('o'*40)
    print('Temperature')
    print(model_T['stamod'].var_exp)
    print('--'*20)
    print('Specific humidity')
    print(model_Q['stamod'].var_exp)
    print('--'*20)
    print('Zonal wind speed')
    print(model_U['stamod'].var_exp)
    print('--'*20)
    print('Meridional wind speed')
    print(model_V['stamod'].var_exp)
    print('--'*20)

def get_attributes(path):
    """
    Get all the attributes at each stations

    :return:
    """
    # Set attributes
    #     AttSta = att_sta('/home/thomas/phd/local_downscaling/data/topo_indexes/df_topoindex/df_topoindex.csv')
    #     AttSta.attributes.dropna(how='all',axis=1, inplace=True)
    #     AttSta.attributes.dropna(how='any',axis=0, inplace=True)
    #     print AttSta.attributes

    # Measured attributed
    AttSta = Att_Sta(str(path  / 'database/predictors/out/att_sta/df_topoindex_ribeirao.csv'))
    AttSta.attributes['Alt'] = AttSta.attributes['Alt'].astype(float) # if not give problem in the stepwise libnear regression
    AttSta.attributes.dropna(how='all',axis=1, inplace=True)
    AttSta.attributes.dropna(how='any',axis=0, inplace=True)

    # remove duplicated attributes
    AttSta.attributes = AttSta.attributes.loc[:,~AttSta.attributes.columns.duplicated()]


    # DEM extracted attributes
    AttSta_topex = Att_Sta( str(path /  'database/predictors/out/att_sta/df_topex_ribeirao.csv'))
    AttSta_topex.attributes = AttSta_topex.attributes.apply(pd.to_numeric, errors='ignore', axis=1)
    AttSta.addatt(df= AttSta_topex.attributes)

    # remove duplicated attributes
    AttSta.attributes = AttSta.attributes.loc[:,~AttSta.attributes.columns.duplicated()]
    AttSta_dist_side = Att_Sta(str(path /  'database/predictors/out/att_sta/df_ribeirao_dist_side.csv'))
    # AttSta_dist_outlet.attributes = AttSta_dist_outlet.attributes.T
    AttSta_dist_side.attributes = AttSta_dist_side.attributes.apply(pd.to_numeric, errors='ignore', axis=1)
    AttSta.addatt(df= AttSta_dist_side.attributes)

    # remove duplicated attributes
    AttSta.attributes = AttSta.attributes.loc[:,~AttSta.attributes.columns.duplicated()]

    AttSta_dist_outlet = Att_Sta(str(path / 'database/predictors/out/att_sta/df_ribeirao_dist_outlet.csv'))
    # AttSta_dist_outlet.attributes = AttSta_dist_outlet.attributes.T
    AttSta_dist_outlet.attributes = AttSta_dist_outlet.attributes.apply(pd.to_numeric, errors='ignore', axis=1)
    AttSta.addatt(df= AttSta_dist_outlet.attributes)


    AttSta_dist_river = Att_Sta(str(path / 'database/predictors/out/att_sta/df_ribeirao_dist_river.csv'))
    # AttSta_dist_river.attributes = AttSta_dist_river.attributes.T
    AttSta_dist_river.attributes = AttSta_dist_river.attributes.apply(pd.to_numeric, errors='ignore', axis=1)
    AttSta.addatt(df= AttSta_dist_river.attributes)

    # mapping for new station database
    # AttSta.attributes.index = ['rib_'+col for col in AttSta.attributes.index]


    AttSta.attributes.index = ['rib_'+sta for sta in  AttSta.attributes.index]

    return AttSta

def get_train_test_index_columns(df, df_gcm):
    """"
    Get train and test data set with Attributes
    """
#===============================================================================
# Create train and testy dataframe
#===============================================================================
#------------------------------------------------------------------------------
# Select same index
#------------------------------------------------------------------------------

    df = df[df.index.isin(df_gcm.index)]
    df_gcm = df_gcm[df_gcm.index.isin(df.index)]
#
# #------------------------------------------------------------------------------
# # train dataset
# #------------------------------------------------------------------------------
#     df_train = df.iloc[:-int(len(df.index)/4),:]
#     df_gfs_train = df_gfs[df_gfs.index.isin(df_train.index)]
#
# # #------------------------------------------------------------------------------
# # # testy dataset
# # #------------------------------------------------------------------------------
#     df_test = df.iloc[-int(len(df.index)/4):,:]
#     df_gfs_test = df_gfs[df_gfs.index.isin(df_test.index)]
#     df_test = df_test[df_test.index.isin(df_gfs_test.index)]

    df_train, df_test = train_test_split(df, test_size=0.20, random_state=5) # !!!!


    return df_train.index, df_test.index, df.index

def plot_loadings(model_T, model_Q, model_U, model_V):
    """
    plot loadings article II

    :return:
    """

    AttSta = get_attributes(path)

    # Plot loadings
    plt.close('all')
    f, ((ax1, ax2,ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10,ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4, figsize=(14, 14))
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax10, ax11, ax12, ax13, ax14, ax15, ax16]

    for ax, pc_nb  in zip([ax1,ax5,ax9,ax13], [1,2,3,4]):
        print(pc_nb)
        x = AttSta.attributes.loc[model_T['stamod'].df.columns, model_T['df_models_loadings'].loc[pc_nb,"predictor"]] # Get the attributes
        y = model_T['stamod'].eigenvectors.loc[pc_nb,:] # Get the loadings
        ax.scatter(x,y,c=blue,s=10) # Plot loading in function of attributes
        ax.plot(np.linspace(x.min(), x.max(),100), model_T['df_models_loadings'].loc[pc_nb,"model"](np.linspace(x.min(), x.max(),100), *model_T['df_models_loadings'].loc[pc_nb, "params"]),c='k') # Plot fit of the attributes
        ax.set_xlabel('$'+model_T['df_models_loadings'].loc[pc_nb,"predictor"]+'$')
        plt.setp( ax.xaxis.get_majorticklabels())


    for ax, pc_nb  in zip([ax2,ax6], [1,2]):
        print(pc_nb)
        x = AttSta.attributes.loc[model_Q['stamod'].df.columns, model_Q['df_models_loadings'].loc[pc_nb,"predictor"]] # Get the attributes
        y = model_Q['stamod'].eigenvectors.loc[pc_nb,:] # Get the loadings
        ax.scatter(x,y,s=10) # Plot loading in function of attributes
        ax.plot(np.linspace(x.min(), x.max(),100), model_Q['df_models_loadings'].loc[pc_nb,"model"](np.linspace(x.min(), x.max(),100), *model_Q['df_models_loadings'].loc[pc_nb, "params"]),c='k') # Plot fit of the attributes
        ax.set_xlabel('$'+model_Q['df_models_loadings'].loc[pc_nb,"predictor"]+'$')
        plt.setp( ax.xaxis.get_majorticklabels())

    for ax, pc_nb  in zip([ax3,ax7], [1,2]):
        print(pc_nb)
        x = AttSta.attributes.loc[model_V['stamod'].df.columns, model_V['df_models_loadings'].loc[pc_nb,"predictor"]] # Get the attributes
        y = model_V['stamod'].eigenvectors.loc[pc_nb,:] # Get the loadings
        ax.scatter(x,y,s=10) # Plot loading in function of attributes
        ax.plot(np.linspace(x.min(), x.max(),100), model_V['df_models_loadings'].loc[pc_nb,"model"](np.linspace(x.min(), x.max(),100), *model_V['df_models_loadings'].loc[pc_nb, "params"]),c='k') # Plot fit of the attributes
        ax.set_xlabel('$'+model_V['df_models_loadings'].loc[pc_nb,"predictor"]+'$')
        plt.setp( ax.xaxis.get_majorticklabels())

    for ax, pc_nb  in zip([ax4,ax8, ax12], [1,2]):
        print(pc_nb)
        x = AttSta.attributes.loc[model_U['stamod'].df.columns, model_U['df_models_loadings'].loc[pc_nb,"predictor"]] # Get the attributes
        y = model_U['stamod'].eigenvectors.loc[pc_nb,:] # Get the loadings
        ax.scatter(x,y,s=10) # Plot loading in function of attributes
        ax.plot(np.linspace(x.min(), x.max(),100), model_U['df_models_loadings'].loc[pc_nb,"model"](np.linspace(x.min(), x.max(),100), *model_U['df_models_loadings'].loc[pc_nb, "params"]),c='k') # Plot fit of the attributes
        ax.set_xlabel('$'+model_U['df_models_loadings'].loc[pc_nb,"predictor"]+'$')
        plt.setp( ax.xaxis.get_majorticklabels())




    ax1.set_title('Temperature ($^\circ C$)')
    ax2.set_title('Specific humidity ($g.kg^{-1}$)')
    ax3.set_title('Meridional wind ($m.s^{-1}$)')
    ax4.set_title('Zonal wind ($m.s^{-1}$)')

    ax1.set_ylabel('$PC_1$')
    ax5.set_ylabel('$PC_2$')
    ax9.set_ylabel('$PC_3$')
    ax13.set_ylabel('$PC_4$')

    for ax in [ax10, ax14, ax15, ax16,ax11]:
        f.delaxes(ax)

    # f.tight_layout(pad=-1)

    for t, ax in zip(["a)","b)",'c)','d)','e)', 'f)', 'g)','h)','i)','j)','k)','l)','m)','n)'],[ax1, ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax11,ax13,ax14]):
        ax.text(-0.10, 0.95, t, transform=ax.transAxes, size=14, weight='bold')

    # plt.tight_layout()
    plt.savefig("/home/thomas/pro/research/framework/res/articleII/loadings_fits_articleII.pdf")
    plt.close('all')
    return axs

def plot_daily_scores_articleII(stamod_hourly_T, stamod_hourly_Q, stamod_hourly_U, stamod_hourly_V, stamod_T, stamod_Q, stamod_U, stamod_V):
    inpath = "/home/thomas/phd/framework/model/out/local_downscaling/ribeirao_articleII/"

    f, ((ax1, ax2,ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10,ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4, figsize=(14, 7))
    # plt.xticks(rotation=70)
    # ax1.plot(stamod_T.scores.loc[:,1].sort_index(), c='0.7')
    # ax5.plot(stamod_T.scores.loc[:,2].sort_index(), c='0.7')
    # ax9.plot(stamod_T.scores.loc[:,3].sort_index(), c='0.7')
    # ax13.plot(stamod_T.scores.loc[:,4].sort_index(), c='0.7')
    #
    # ax2.plot(stamod_Q.scores.loc[:,1].sort_index(), c='0.7')
    # ax6.plot(stamod_Q.scores.loc[:,2].sort_index(), c='0.7')
    #
    # ax3.plot(stamod_U.scores.loc[:,1].sort_index(), c='0.7')
    # ax7.plot(stamod_U.scores.loc[:,2].sort_index(), c='0.7')
    # # ax11.plot(stamod_U.scores.loc[:,3], c='0.7')
    # # ax15.plot(stamod_U.scores.loc[:,4], c='0.7')
    #
    # ax4.plot(stamod_V.scores.loc[:,1].sort_index(), c='0.7')
    # ax8.plot(stamod_V.scores.loc[:,2].sort_index(), c='0.7')
    # ax12.plot(stamod_V.scores.loc[:,3].sort_index(), c='0.7')
    # # ax16.plot(stamod_V.scores.loc[:,4], c='0.7')

    # HOURLY
    t1 = stamod_hourly_T.scores.loc[:,1].sort_index().groupby(lambda x:x.hour).mean()*-1
    t1.plot(ax=ax1, c='0.7', lw =3)

    t2 = stamod_hourly_T.scores.loc[:,2].sort_index().groupby(lambda x:x.hour).mean() *-1
    t2.plot(ax=ax5, c='0.7', lw =3)

    t3 = stamod_hourly_T.scores.loc[:,3].sort_index().groupby(lambda x:x.hour).mean()
    t3.plot(ax=ax9, c='0.7', lw =3)

    t4 = stamod_hourly_T.scores.loc[:,4].sort_index().groupby(lambda x:x.hour).mean()*-1
    t4.plot(ax=ax13, c='0.7', lw =3)

    q1 = stamod_hourly_Q.scores.loc[:,1].sort_index().groupby(lambda x:x.hour).mean()
    q1.plot(ax=ax2, c='0.7', lw =3)

    q2 = stamod_hourly_Q.scores.loc[:,2].sort_index().groupby(lambda x:x.hour).mean() *-1
    q2.plot(ax=ax6, c='0.7', lw =3)

    u1 = stamod_hourly_U.scores.loc[:,1].sort_index().groupby(lambda x:x.hour).mean()*-1
    u1.plot(ax=ax3,  c='0.7', lw =3)

    u2 = stamod_hourly_U.scores.loc[:,2].sort_index().groupby(lambda x:x.hour).mean()
    u2.plot(ax=ax7, c='0.7', lw =3)

    v1 = stamod_hourly_V.scores.loc[:,1].sort_index().groupby(lambda x:x.hour).mean()*-1
    v1.plot(ax=ax4, c='0.7', lw =3)

    v2 = stamod_hourly_V.scores.loc[:,2].sort_index().groupby(lambda x:x.hour).mean()
    v2.plot(ax=ax8, c='0.7', lw =3)
    # stamod_hourly_V.scores.loc[:,3].sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax12, c='0.7', lw =3)

    # 6 HOURLY - ZZZ
    pd.DataFrame(stamod_T['res']['lin']['scores'][0,:,0], index=stamod_T['res']['lin']['index']).sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax1, color='0.5',linestyle="None", marker='o', alpha=0.5, legend=False)
    pd.DataFrame(stamod_T['res']['lin']['scores'][0,:,1], index=stamod_T['res']['lin']['index']).sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax5, color='0.5',linestyle="None",marker='o', alpha=0.5, legend=False)
    pd.DataFrame(stamod_T['res']['lin']['scores'][0,:,2], index=stamod_T['res']['lin']['index']).sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax9, color='0.5',marker='o',linestyle="None", alpha=0.5, legend=False)
    pd.DataFrame(stamod_T['res']['lin']['scores'][0,:,3], index=stamod_T['res']['lin']['index']).sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax13,color='0.5', marker='o',linestyle="None", alpha=0.5, legend=False)
    #
    pd.DataFrame(stamod_Q['res']['lin']['scores'][0,:,0], index=stamod_Q['res']['lin']['index']).sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax2,color='0.5', marker='o',linestyle="None", alpha=0.5, legend=False)
    pd.DataFrame(stamod_Q['res']['lin']['scores'][0,:,1], index=stamod_Q['res']['lin']['index']).sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax6,color='0.5', marker='o',linestyle="None", alpha=0.5, legend=False)

    pd.DataFrame(stamod_U['res']['lin']['scores'][0,:,0], index=stamod_U['res']['lin']['index']).sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax3,color='0.5', marker='o', alpha=0.5,linestyle="None", legend=False)
    pd.DataFrame(stamod_U['res']['lin']['scores'][0,:,1], index=stamod_U['res']['lin']['index']).sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax7,color='0.5', marker='o', alpha=0.5,linestyle="None", legend=False)

    pd.DataFrame(stamod_V['res']['lin']['scores'][0,:,0], index=stamod_V['res']['lin']['index']).sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax4, color='0.5',marker='o', alpha=0.5,linestyle="None", legend=False)
    pd.DataFrame(stamod_V['res']['lin']['scores'][0,:,1], index=stamod_V['res']['lin']['index']).sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax8, color='0.5',marker='o', alpha=0.5,linestyle="None", legend=False)


    # 6 HOURLY - OBSERVED
    stamod_T['stamod'].scores.loc[:,1].sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax1, style='k^', alpha=0.5, legend=False)
    stamod_T['stamod'].scores.loc[:,2].sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax5,style='k^', alpha=0.5, legend=False)
    stamod_T['stamod'].scores.loc[:,3].sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax9, style='k^', alpha=0.5, legend=False)
    stamod_T['stamod'].scores.loc[:,4].sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax13, style='k^', alpha=0.5, legend=False)

    stamod_Q['stamod'].scores.loc[:,1].sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax2, style='k^', alpha=0.5, legend=False)
    stamod_Q['stamod'].scores.loc[:,2].sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax6,  style='k^', alpha=0.5, legend=False)

    stamod_U['stamod'].scores.loc[:,1].sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax3, style='k^', alpha=0.5, legend=False)
    stamod_U['stamod'].scores.loc[:,2].sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax7, style='k^', alpha=0.5, legend=False)
    # ax11.plot(stamod_U.scores.loc[:,3], c='0.7')
    # ax15.plot(stamod_U.scores.loc[:,4], c='0.7')

    stamod_V['stamod'].scores.loc[:,1].sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax4, style='k^', alpha=0.5, legend=False)
    stamod_V['stamod'].scores.loc[:,2].sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax8, style='k^', alpha=0.5, legend=False)
    # stamod_V.scores.loc[:,3].sort_index().groupby(lambda x:x.hour).mean().plot(ax=ax12, style='k^')


    # ax16.plot(stamod_V.scores.loc[:,4], c='0.7')

    ax1.tick_params(labelbottom='off')
    ax5.tick_params(labelbottom='off')
    ax9.tick_params(labelbottom='off')

    ax2.tick_params(labelbottom='off')

    ax3.tick_params(labelbottom='off')

    ax4.tick_params(labelbottom='off')
    ax8.tick_params(labelbottom='off')


    for ax in [ax10, ax14,ax11, ax15, ax16, ax12]:
        f.delaxes(ax)

    ax1.set_title('Temperature ($^\circ C$)')
    ax2.set_title('Specific humidity ($g.kg^{-1}$)')
    ax3.set_title('Zonal wind ($m.s^{-1}$)')
    ax4.set_title('Meridional wind ($m.s^{-1}$)')

    ax1.set_ylabel('$PC_1$ scores')
    ax5.set_ylabel('$PC_2$ scores')
    ax9.set_ylabel('$PC_3$ scores')
    ax13.set_ylabel('$PC_4$ scores')

    axs = [ax1, ax2, ax3,ax4, ax5, ax6, ax7,ax8,ax9,ax12,ax13]
    for t, ax in zip(["a)","b)",'c)','d)','e)', 'f)', 'g)','h)','i)','j)','k)','l)'],axs):
        ax.grid(True)
        ax.text(-0.10, 1, t, transform=ax.transAxes, size=12, weight='bold')
        ax.set_xticks([3,9,15,21])
        ax.set_xticklabels  ([3,9,15,21])

    # plt.tight_layout()

    plt.savefig('/home/thomas/pro/research/framework/res/articleII/daily_scores.pdf')
    plt.close('all')

def plot_scores_articleII(stamod_T, stamod_Q, stamod_U, stamod_V):
    inpath = "/home/thomas/phd/framework/model/out/local_downscaling/ribeirao_articleII/"

    f, ((ax1, ax2,ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10,ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4, figsize=(14, 7))

    stamod_T.scores.loc[:,1].sort_index().plot(ax=ax1, c='0.7')
    stamod_T.scores.loc[:,1].sort_index().rolling(40).mean().plot(ax=ax1, c='0.3', linewidth=3, rot=0)

    stamod_T.scores.loc[:,2].sort_index().plot(ax=ax5, c='0.7')
    stamod_T.scores.loc[:,2].sort_index().rolling(40).mean().plot(ax=ax5, c='0.3', linewidth=3, rot=0)

    stamod_T.scores.loc[:,3].sort_index().plot(ax=ax9, c='0.7')
    stamod_T.scores.loc[:,3].sort_index().rolling(40).mean().plot(ax=ax9, c='0.3', linewidth=3, rot=0)

    stamod_T.scores.loc[:,4].sort_index().plot(ax=ax13, c='0.7')
    stamod_T.scores.loc[:,4].sort_index().rolling(40).mean().plot(ax=ax13, c='0.3', linewidth=3, rot=0)

    stamod_Q.scores.loc[:,1].sort_index().plot(ax=ax2, c='0.7')
    stamod_Q.scores.loc[:,1].sort_index().rolling(40).mean().plot(ax=ax2, c='0.3', linewidth=3, rot=0)

    stamod_Q.scores.loc[:,2].sort_index().plot(ax=ax6, c='0.7')
    stamod_Q.scores.loc[:,2].sort_index().rolling(40).mean().plot(ax=ax6, c='0.3', linewidth=3, rot=0)

    stamod_U.scores.loc[:,1].sort_index().plot(ax=ax3, c='0.7')
    stamod_U.scores.loc[:,1].sort_index().rolling(40).mean().plot(ax=ax3, c='0.3', linewidth=3, rot=0)

    stamod_U.scores.loc[:,2].sort_index().plot(ax=ax7, c='0.7')
    stamod_U.scores.loc[:,2].sort_index().rolling(40).mean().plot(ax=ax7, c='0.3', linewidth=3, rot=0)
    # ax11.plot(stamod_U.scores.loc[:,3], c='0.7')
    # ax15.plot(stamod_U.scores.loc[:,4], c='0.7')

    stamod_V.scores.loc[:,1].sort_index().plot(ax=ax4, c='0.7')
    stamod_V.scores.loc[:,1].sort_index().rolling(60).mean().plot(ax=ax4, c='0.3', linewidth=3, rot=0)

    stamod_V.scores.loc[:,2].sort_index().plot(ax=ax8, c='0.7')
    stamod_V.scores.loc[:,2].sort_index().rolling(60).mean().plot(ax=ax8, c='0.3', linewidth=3, rot=0)

    # stamod_V.scores.loc[:,3].sort_index().plot(ax=ax12, c='0.7')
    # stamod_V.scores.loc[:,3].sort_index().rolling(40).mean().plot(ax=ax12, c='0.3', linewidth=3)
    # ax16.plot(stamod_V.scores.loc[:,4], c='0.7')

    ax1.tick_params(labelbottom='off')
    ax5.tick_params(labelbottom='off')
    ax9.tick_params(labelbottom='off')

    ax2.tick_params(labelbottom='off')

    ax3.tick_params(labelbottom='off')

    # ax4.tick_params(labelbottom='off')
    # ax8.tick_params(labelbottom='off')


    for ax in [ax10, ax14,ax11, ax15, ax16,ax12]:
        f.delaxes(ax)

    ax1.set_title('Temperature ($^\circ C$)')
    ax2.set_title('Specific humidity ($g.kg^{-1}$)')
    ax3.set_title('Zonal wind ($m.s^{-1}$)')
    ax4.set_title('Meridional wind ($m.s^{-1}$)')

    ax1.set_ylabel('PC$_1$ scores')
    ax5.set_ylabel('PC$_2$ scores')
    ax9.set_ylabel('PC$_3$ scores')
    ax13.set_ylabel('PC$_4$ scores')

    # plt.tight_layout()

    axs = [ax1, ax2, ax3,ax4, ax5, ax6, ax7,ax8,ax9, ax13]
    for t, ax in zip(["a)","b)",'c)','d)','e)', 'f)', 'g)','h)','i)','j)','k)','l)'],axs):
        ax.text(-0.10, 0.95, t, transform=ax.transAxes, size=12, weight='bold')
        ax.grid(True, alpha=0.5)



    plt.savefig('/home/thomas/pro/research/framework/res/articleII/scores.png')
    plt.close('all')

def get_csf():
    df_irr = pd.read_csv('/home/thomas/pro/research/framework/stations_database/1_database/data/ribeirao_irr/EXT_Rad_hora.csv', index_col=0, parse_dates=True)
    df_irr.replace(-9999, np.NaN, inplace=True)
    df_irr = df_irr.loc[:,'Pira_369']

    irr = LCB_Irr()
    inpath_sim='/home/thomas/Dropbox/pro/research/framework/predictors/out/irr/Irradiance_rsun_lin2_lag-0.2_glob_df.csv'
    irr.read_sim(inpath_sim)
    df_irr_sim = irr.data_sim.loc[:,'C04']

    years = range(0,4)
    df_irr_sim_years = []
    for year in years:
        df_irr_sim_years.append(pd.DataFrame(df_irr_sim.values, index=df_irr_sim.index + pd.Timedelta(days=365*year)))

    df_irr_sim = pd.concat(df_irr_sim_years,axis=0)
    df_irr_sim = df_irr_sim[~df_irr_sim.index.duplicated(keep='first')]

    df_irr.dropna(axis=0, inplace=True)
    df_irr_sim.dropna(axis=0, inplace=True)

    df = pd.concat([df_irr, df_irr_sim.iloc[:,0]], axis=1,join='outer')
    df.columns = ['obs', 'sim']

    csf = df['obs']/df['sim']
    csf = csf.between_time('8:00','15:00')
    # csf = csf.resample('D').mean()
    # csf = csf.resample('H').pad()
    csf.name ='csf_obs'
    return csf

def dfs_err(res, stamod):

    df_err_models_pcs = []

    for model in res.keys():
        for pc in range(res[model]['predicted'].shape[-1]):
            rec_pc_est = pd.DataFrame(res[model]['predicted'][:, :, pc], index=res[model]['index'],
                                      columns=stamod.eigenvectors.columns)
            rec_pc_obs = stamod.pca_reconstruct()[pc + 1].loc[res[model]['index'],:]
            err = rec_pc_est - rec_pc_obs
            df_err = pd.DataFrame(err.values.flatten(), columns=['err'])
            df_err['model'] = model
            df_err['pc'] = str(pc + 1)
            df_err_models_pcs.append(df_err)

        # total
        est_total = res[model]['predicted'].sum(axis=2)
        obs_total = stamod.df.loc[res[model]['index'], :]
        err_total = est_total - obs_total
        df_err = pd.DataFrame(err_total.values.flatten(), columns=['err'])
        df_err['model'] = model
        df_err['pc'] = 'Total'
        df_err_models_pcs.append(df_err)

    dfs = pd.concat(df_err_models_pcs, axis=0)
    return dfs

def plot_rmse_by_quantile(model_T, model_Q, model_U, model_V, df_gcm_t, df_gcm_q, df_gcm_u, df_gcm_v,remove_mean=False):

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    titles = [r'Temperature ($^\circ C$)', r'Specific humidity ($g.kg^{-1}$)', r'Zonal wind speed ($m.s^{-1}$)',r'Meridional wind speed ($m.s^{-1}$)']


    # quantiles = np.array([0,0.05,0.95,1])
    quantiles = np.array([0,0.1, 0.9,1])
    # quantiles = np.arange(0,1,0.15)

    for ax, model,title, df_gcm in zip(axes, [model_T, model_Q, model_U, model_V], titles, [df_gcm_t, df_gcm_q, df_gcm_u, df_gcm_v]):

        res = model['res']
        dfs = {}
        # pcs = [1,2]
        # linestyles = ['-','--',':','-.']
        colors = []
        #
        # for model, remove_mean in zip(model,remove_means):
        for nb_pc in model['stamod'].scores.columns[:2]:
            colors.extend(['#002699','#80bfff','#ffa31a','0.3','#00b3b3'])

            for model_type in res.keys():

                df_obs = model['stamod'].df.loc[res[model_type]['index'], :]
                df_pred = pd.DataFrame(res[model_type]['predicted'].sum(axis=2), index=df_obs.index, columns = df_obs.columns)

                df_obs = df_obs.loc[:,'rib_C10']
                df_pred = df_pred.loc[:,'rib_C10']

                if remove_mean:
                    df_obs = model['stamod'].df.loc[res[model_type]['index'], :].add(df_gcm.loc[res[model_type]['index']],axis=0)
                    df_pred = pd.DataFrame(res[model_type]['predicted'].sum(axis=2), index=df_obs.index,columns=df_obs.columns).add(df_gcm.loc[res[model_type]['index']],axis=0)

                scores = model['stamod'].scores[nb_pc]
                scores = scores.loc[res[model_type]['index']]
                values_quantiles = np.quantile(scores, quantiles)


                # df_obs_std = df_obs.loc[res[model_type]['index']].std(axis=1)
                # df_obs_std = df_obs.loc[res[model_type]['index']].max(axis=1) - df_obs.loc[res[model_type]['index']].min(axis=1)
                # values_quantiles = np.quantile(df_obs_std, quantiles)


                dfs_models = []
                qs = []
                for qmin, qmax, quantile_min, quantile_max in zip(values_quantiles[:-1], values_quantiles[1:],
                                                                  quantiles[:-1], quantiles[1:]):
                    # idx = df_obs_std.between(qmin,qmax)
                    idx = scores.between(qmin,qmax)
                    mse = metrics.mean_squared_error(df_obs[idx].values, df_pred[idx].values)
                    rmse= np.power(mse,1./2)
                    dfs_models.append(rmse)
                    qs.append(np.mean([quantile_min, quantile_max]))
                dfs[model_type+' PC'+str(nb_pc)] = dfs_models

                # ax.plot(np.array(qs),np.array(dfs_models), label=model_type, linewidth = 2, linestyle='-',marker='o',markersize=6,color=c)

            if df_gcm is not None:
                df = model['stamod'].df
                df_obs = df.loc[res['lin']['index'], :]

                if remove_mean:
                    df_obs = model['stamod'].df.loc[res['lin']['index'], :].add(df_gcm.loc[res['lin']['index']],axis=0)

                df_gcm = df_gcm.loc[res['lin']['index']]
                df_gcm_merged = pd.concat([df_gcm for sta in df_obs.columns], axis=1)
                df_gcm_merged.columns = df_obs.columns


                scores = model['stamod'].scores[nb_pc]
                scores = scores.loc[res[model_type]['index']]
                values_quantiles = np.quantile(scores, quantiles)

                df_obs = df_obs.loc[:,'rib_C10']
                df_gcm_merged = df_gcm_merged.loc[:,'rib_C10']

                # df_obs_std = df_obs.loc[res['lin']['index']].std(axis=1)
                # df_obs_std = df_obs.loc[res[model_type]['index']].max(axis=1) - df_obs.loc[res[model_type]['index']].min(axis=1)
                # values_quantiles = np.quantile(df_obs_std, quantiles)


                dfs_raw = []
                qs = []
                for qmin, qmax, quantile_min, quantile_max in zip(values_quantiles[:-1], values_quantiles[1:], quantiles[:-1], quantiles[1:]):
                    # idx = df_obs_std.between(qmin,qmax)
                    idx = scores.between(qmin,qmax)

                    mse = metrics.mean_squared_error(df_obs[idx].values, df_gcm_merged[idx].values)
                    rmse= np.power(mse,1./2)

                    dfs_raw.append(rmse)
                    qs.append(np.mean([quantile_min, quantile_max]))
                dfs['raw gcm PC'+str(nb_pc)] = dfs_raw

                ax.plot(np.array(qs), np.array(dfs_raw), label='interp', linewidth = 2, linestyle='--',marker='o',color='k',markersize=6)


            obs_scores=True
            if obs_scores is not None:
                df = model['stamod'].df
                df_obs = df.loc[res['lin']['index'], :]

                # if remove_mean:
                #     df_obs = model['stamod'].df.loc[res['lin']['index'], :].add(df_gcm.loc[res['lin']['index']],axis=0)

                dic_rec = model['stamod'].pca_reconstruct()
                dfs_rec = []
                for r in dic_rec.keys():
                    dfs_rec.append(pd.DataFrame(dic_rec[r]))
                dfs_rec = reduce(lambda x, y: x.add(y, fill_value=0), dfs_rec)
                dfs_rec = dfs_rec.loc[res['lin']['index'],:]


                df_obs = df_obs.loc[:,'rib_C10']
                dfs_rec = dfs_rec.loc[:,'rib_C10']


                scores = model['stamod'].scores[nb_pc]
                scores = scores.loc[res[model_type]['index']]
                values_quantiles = np.quantile(scores, quantiles)

                # df_obs_std = df_obs.loc[res['lin']['index']].std(axis=1)
                # df_obs_std = df_obs.loc[res[model_type]['index']].max(axis=1) - df_obs.loc[res[model_type]['index']].min(axis=1)
                # values_quantiles = np.quantile(df_obs_std, quantiles)

                dfs_obs = []
                qs = []
                for qmin, qmax, quantile_min, quantile_max in zip(values_quantiles[:-1], values_quantiles[1:], quantiles[:-1], quantiles[1:]):
                    # idx = df_obs_std.between(qmin,qmax)
                    idx = scores.between(qmin,qmax)

                    mse = metrics.mean_squared_error(df_obs[idx].values, dfs_rec[idx].values)
                    rmse= np.power(mse,1./2)

                    dfs_obs.append(rmse)
                    qs.append(np.mean([quantile_min, quantile_max]))
                dfs['obs scores PC'+str(nb_pc)] = dfs_obs

                ax.plot(np.array(qs), np.array(dfs_obs), label='Obs PCs scores', linewidth = 2, linestyle=':',marker='o',color='k',markersize=6)

        # dfs = pd.DataFrame(dfs)
        # dfs.index = qs
        # dfs = dfs.iloc[[0,2],:] # TODO WARNING TAKE ONLY EXTREMES
        # dfs.plot(kind='bar',ax=ax, colors=colors*3, legend=False, rot=0)



        # ax.set_xticks(qs)
        ax.set_xticklabels([r'$<q_{0.2}$',r'$>q_{0.8}$'], size=10)
        # ax.ticklabel_format(style='sci')

        # ax2 = ax.twiny()
        ax.set_xlabel('Quantiles', size=10)

        # # ax2.set_xlim(ax.get_xlim())
        # ax2.set_xticks(quantiles)
        # ax2.set_xticklabels(np.around(values_quantiles,1))
        ax.set_title(title, size=10)
        ax.grid(True)


    # plt.legend()
    axes[0].legend( prop={'size': 6})
    axes[0].set_ylabel('Mean Squared Error', size=10)
    plt.tight_layout()
    # axes[0].set_title('Mean Squared Error')
    plt.savefig('/home/thomas/pro/research/framework/res/articleII/mae_quantiles_models_com.pdf')
    plt.close('all')

def plot_quantile_pc(model, nb_pc, axes):
    obs_rec = model['stamod'].pca_reconstruct()
    for pc in np.arange(1, nb_pc + 1):
        ax = axes[pc-1]
        res = model['res']
        ax.grid(True, alpha=0.5)
        for model_type in res.keys():
            est = res[model_type]['predicted'][:, :, pc - 1].flatten()
            obs = obs_rec[pc].loc[res[model_type]['index'], :].values.flatten()
            quantiles = np.array(
                [stats.percentileofscore(obs, x) / 100 for x in np.linspace(obs.min(), obs.max(), 50)])
            qest = np.quantile(est, quantiles)
            qobs = np.quantile(obs, quantiles)

            a = np.array([qobs.min(), qobs.max()])
            ax.plot(a, a, color='k', alpha=0.2, linewidth=1)
            ax.scatter(qobs, qest, label=model_type,s=4,alpha=0.8)

def quantiles_plot(model_T, model_Q, model_U, model_V, df_t_gcm,df_q_gcm, df_u_gcm, df_v_gcm, remove_mean=False):
    # Observed quantile against Estimated quantiles

    fig, axes = plt.subplots(3, 4, figsize=(7, 7))

    # Total
    for ax, model, gcm in zip(axes[0,[0,1,2,3]], [model_T, model_Q, model_U, model_V], [df_t_gcm,df_q_gcm, df_u_gcm, df_v_gcm]):
        res = model['res']
        for model_type in res.keys():
            df_obs = model['stamod'].df.loc[res[model_type]['index'], :]
            if remove_mean:
                df_obs = df_obs.add(gcm.loc[res[model_type]['index']], axis=0)
                est = pd.DataFrame(res[model_type]['predicted'].sum(axis=2), index=df_obs.index,columns=df_obs.columns).add(gcm.loc[res[model_type]['index']], axis=0).values.flatten()
            else:
                est = res[model_type]['predicted'].sum(axis=2).flatten()

            obs = df_obs.values.flatten()

            quantiles = np.array([stats.percentileofscore(obs, x) / 100 for x in np.linspace(obs.min(), obs.max(), 50)])
            qest = np.quantile(est, quantiles)
            qobs = np.quantile(obs, quantiles)

            ax.grid(True, alpha=0.5)
            a = np.array([qobs.min(), qobs.max()])
            ax.plot(a, a, color='0.5', alpha=0.5)
            ax.scatter(qobs, qest, label=model_type,s=6,alpha=0.5)


    # Total gfs interp

    for ax, model, gcm in zip(axes[0,[0,1,2,3]], [model_T,model_Q,  model_U, model_V], [df_t_gcm,df_q_gcm, df_u_gcm, df_v_gcm]):

        df_obs = model['stamod'].df.loc[res[model_type]['index'], :]
        if remove_mean:
            df_obs = df_obs.add(gcm.loc[res[model_type]['index']], axis=0)
            est = gcm.loc[res[model_type]['index']].values.flatten()

        else:
            est = gcm.loc[res[model_type]['index']].values.flatten()
        obs = df_obs.values.flatten()



        quantiles = np.array([stats.percentileofscore(obs, x) / 100 for x in np.linspace(obs.min(), obs.max(), 50)])
        qest = np.quantile(est, quantiles)
        qobs = np.quantile(obs, quantiles)
        a = np.array([qobs.min(), qobs.max()])
        ax.plot(a, a, color='0.5', alpha=0.5)
        ax.grid(True, alpha=0.5)
        ax.scatter(qobs, qest, label='interp', s=6,color='k',alpha=0.5)

    # PC temperature
    plot_quantile_pc(model_T, 2, axes[1:,0])
    plot_quantile_pc(model_Q, 2, axes[1:,1])
    plot_quantile_pc(model_U, 2, axes[1:,2])
    plot_quantile_pc(model_V, 2, axes[1:,3])

    axes[0,0].set_title(r'Temperature ($^\circ C$)', size=8)
    axes[0,1].set_title(r'Specific humidity ($g.kg^{-1}$)', size=8)
    axes[0,2].set_title(r'Zonal wind speed ($m.s^{-1}$)', size=8)
    axes[0,3].set_title(r'Meridional wind speed ($m.s^{-1}$)', size=8)

    # axes[0,0].set_ylabel('Total')
    axes[1,0].set_ylabel(r'PC$_1$',size=8)
    axes[2,0].set_ylabel(r'PC$_2$',size=8)

    axes[0,0].legend( prop={'size': 6})
    # plt.tight_layout()
    plt.savefig('/home/thomas/pro/research/framework/res/articleII/quantiles_models_com.pdf')
    plt.close('all')
    # plt.show()

def plot_field_var(xr, filename):
    f, (
    (ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(4,4, figsize=(10, 8))

    xr_9 = xr['clim'].where(xr.time.dt.hour == 9).mean(dim='time').where(xr.mask > 0, drop=True)
    xr_15 = xr['clim'].where(xr.time.dt.hour == 15).mean(dim='time').where(xr.mask > 0, drop=True)
    xr_21 = xr['clim'].where(xr.time.dt.hour == 21).mean(dim='time').where(xr.mask > 0, drop=True)
    xr_3 = xr['clim'].where(xr.time.dt.hour == 3).mean(dim='time').where(xr.mask > 0, drop=True)


        # Tlim = np.linspace(xr['clim'].sel(var='T').where(xr.mask > 0).min(),xr['clim'].sel(var='T').where(xr.mask > 0, drop=True).max(),10)
        # Qlim = np.linspace(xr['clim'].sel(var='Q').where(xr.mask > 0).min(),xr['clim'].sel(var='T').where(xr.mask > 0, drop=True).max(),10)
        # Ulim = np.linspace(xr['clim'].sel(var='U').where(xr.mask > 0).min(),xr['clim'].sel(var='T').where(xr.mask > 0, drop=True).max(),10)
        # Vlim = np.linspace(xr['clim'].sel(var='V').where(xr.mask > 0).min(),xr['clim'].sel(var='T').where(xr.mask > 0, drop=True).max(),10)

    xr_9.sel(var='T').plot(ax=ax1, cmap='RdBu_r')
    xr_15.sel(var='T').plot(ax=ax5, cmap='RdBu_r')
    xr_21.sel(var='T').plot(ax=ax9, cmap='RdBu_r')
    xr_3.sel(var='T').plot(ax=ax13, cmap='RdBu_r')

    xr_9.sel(var='Q').plot(ax=ax2, cmap='RdBu_r')
    xr_15.sel(var='Q').plot(ax=ax6, cmap='RdBu_r')
    xr_21.sel(var='Q').plot(ax=ax10, cmap='RdBu_r')
    xr_3.sel(var='Q').plot(ax=ax14, cmap='RdBu_r')

    xr_9.sel(var='U').plot(ax=ax3, cmap='RdBu_r')
    xr_15.sel(var='U').plot(ax=ax7, cmap='RdBu_r')
    xr_21.sel(var='U').plot(ax=ax11, cmap='RdBu_r')
    xr_3.sel(var='U').plot(ax=ax15, cmap='RdBu_r')

    xr_9.sel(var='V').plot(ax=ax4, cmap='RdBu_r')
    xr_15.sel(var='V').plot(ax=ax8, cmap='RdBu_r')
    xr_21.sel(var='V').plot(ax=ax12, cmap='RdBu_r')
    xr_3.sel(var='V').plot(ax=ax16, cmap='RdBu_r')

    ax1.set_ylabel('9 H')
    ax5.set_ylabel('15 H')
    ax9.set_ylabel('21 H')
    ax13.set_ylabel('3 H')

    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16]
    for t, ax in zip(["a)", "b)", 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)', 'm)'], axs):
        ax.text(-0.15, 0.95, t, transform=ax.transAxes, size=12, weight='bold')

    for ax in axs:
        ax.set_title('')

    ax1.set_title(r'Temperature ($^\circ C$)',size=10)
    ax2.set_title(r'Specific humidity ($g.kg^{-1}$)',size=10)
    ax3.set_title(r'Zonal wind ($m.s^{-1}$)',size=10)
    ax4.set_title(r'Meridional wind($m.s^{-1}$)',size=10)

    ax1.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax4.get_xaxis().set_visible(False)
    ax5.get_xaxis().set_visible(False)
    ax6.get_xaxis().set_visible(False)
    ax7.get_xaxis().set_visible(False)
    ax8.get_xaxis().set_visible(False)
    ax9.get_xaxis().set_visible(False)
    ax10.get_xaxis().set_visible(False)
    ax11.get_xaxis().set_visible(False)
    ax12.get_xaxis().set_visible(False)

    ax2.get_yaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)
    ax7.get_yaxis().set_visible(False)
    ax8.get_yaxis().set_visible(False)
    ax10.get_yaxis().set_visible(False)
    ax11.get_yaxis().set_visible(False)
    ax12.get_yaxis().set_visible(False)
    ax14.get_yaxis().set_visible(False)
    ax15.get_yaxis().set_visible(False)
    ax16.get_yaxis().set_visible(False)

    # plt.tight_layout()
    plt.savefig('/home/thomas/pro/research/framework/res/articleII/' + filename)
    plt.close('all')

def plot_fields_spatial_predictors(xr_loadings):
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(8, 8))

    xr_loadings.topex_ne.where(xr_loadings.mask > 0,drop=True).plot(ax=ax1, cmap='RdBu_r')
    xr_loadings.topex_w.where(xr_loadings.mask > 0,drop=True).plot(ax=ax2, cmap='RdBu_r')
    xr_loadings.Alt.where(xr_loadings.mask > 0,drop=True).plot(ax=ax3, cmap='RdBu_r')
    xr_loadings.dist_outlet.where(xr_loadings.mask > 0,drop=True).plot(ax=ax4, cmap='RdBu_r')
    xr_loadings.dist_river.where(xr_loadings.mask > 0,drop=True).plot(ax=ax5, cmap='RdBu_r')
    xr_loadings.dist_side.where(xr_loadings.mask > 0,drop=True).plot(ax=ax6, cmap='RdBu_r')

    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    for t, ax in zip(["a)", "b)", 'c)', 'd)', 'e)', 'f)'], axs):
        ax.text(-0.15, 1, t, transform=ax.transAxes, size=10, weight='bold')
        ax.ticklabel_format(axis='y', style='sci')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    titles = ['Topex NE', 'Topex W','Alt', 'Dist. Outlet', 'Dist. river', 'Dist. sidewall']

    for ax, title in zip(axs, titles):
        ax.set_title(title, size=10)


    ax1.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax4.get_xaxis().set_visible(False)

    ax2.get_yaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)

    # plt.tight_layout()
    plt.savefig('/home/thomas/pro/research/framework/res/articleII/field_predictors.png')

def plot_field_loadings(xr_loadings):
    f, (
    (ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(4,4, figsize=(10, 8))

    xr_loadings['loadings'].sel(var='T', pc_nb=1).where(xr_loadings.mask > 0, drop=True).plot(ax=ax1, cmap='RdBu_r', cbar_kwargs={'format':'%.0e','label':''})
    xr_loadings['loadings'].sel(var='T', pc_nb=2).where(xr_loadings.mask > 0, drop=True).plot(ax=ax5, cmap='RdBu_r', cbar_kwargs={'format':'%.0e','label':''})
    xr_loadings['loadings'].sel(var='T', pc_nb=3).where(xr_loadings.mask > 0, drop=True).plot(ax=ax9, cmap='RdBu_r', cbar_kwargs={'format':'%.0e','label':''})
    xr_loadings['loadings'].sel(var='T', pc_nb=4).where(xr_loadings.mask > 0, drop=True).plot(ax=ax13, cmap='RdBu_r', cbar_kwargs={'format':'%.0e','label':''})

    xr_loadings['loadings'].sel(var='Q', pc_nb=1).where(xr_loadings.mask > 0, drop=True).plot(ax=ax2, cmap='RdBu_r', cbar_kwargs={'format':'%.0e','label':''})
    xr_loadings['loadings'].sel(var='Q', pc_nb=2).where(xr_loadings.mask > 0, drop=True).plot(ax=ax6, cmap='RdBu_r', cbar_kwargs={'format':'%.0e','label':''})

    xr_loadings['loadings'].sel(var='U', pc_nb=1).where(xr_loadings.mask > 0, drop=True).plot(ax=ax3, cmap='RdBu_r', cbar_kwargs={'format':'%.0e','label':''})
    xr_loadings['loadings'].sel(var='U', pc_nb=1).where(xr_loadings.mask > 0, drop=True).plot(ax=ax7, cmap='RdBu_r', cbar_kwargs={'format':'%.0e','label':''})

    xr_loadings['loadings'].sel(var='V', pc_nb=1).where(xr_loadings.mask > 0, drop=True).plot(ax=ax4, cmap='RdBu_r', cbar_kwargs={'format':'%.0e','label':''})
    xr_loadings['loadings'].sel(var='V', pc_nb=2).where(xr_loadings.mask > 0, drop=True).plot(ax=ax8, cmap='RdBu_r', cbar_kwargs={'format':'%.0e','label':''})
    # xr_loadings['loadings'].sel(var='V', pc_nb=3).where(xr_loadings.mask > 0, drop=True).plot(ax=ax12, cmap='RdBu_r')

    ax1.set_ylabel('PC1')
    ax5.set_ylabel('PC2')
    ax9.set_ylabel('PC3')
    ax13.set_ylabel('PC4')

    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11,  ax13, ax14, ax15, ax16]
    for t, ax in zip(["a)", "b)", 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)'], axs):
        ax.text(-0.25, 0.95, t, transform=ax.transAxes, size=11, weight='bold')


    for ax in axs:
        ax.set_title('')

    ax1.set_title(r'Temperature ($^\circ C$)',size=10)
    ax2.set_title(r'Specific humidity ($g.kg^{-1}$)',size=10)
    ax3.set_title(r'Zonal wind ($m.s^{-1}$)',size=10)
    ax4.set_title(r'Meridional wind($m.s^{-1}$)',size=10)

    ax1.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax4.get_xaxis().set_visible(False)
    ax5.get_xaxis().set_visible(False)
    ax6.get_xaxis().set_visible(False)
    ax7.get_xaxis().set_visible(False)
    ax8.get_xaxis().set_visible(False)
    ax9.get_xaxis().set_visible(False)
    ax10.get_xaxis().set_visible(False)
    ax11.get_xaxis().set_visible(False)
    # ax12.get_xaxis().set_visible(False)

    ax2.get_yaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)
    ax7.get_yaxis().set_visible(False)
    ax8.get_yaxis().set_visible(False)
    ax10.get_yaxis().set_visible(False)
    ax11.get_yaxis().set_visible(False)
    # ax12.get_yaxis().set_visible(False)
    ax14.get_yaxis().set_visible(False)
    ax15.get_yaxis().set_visible(False)
    ax16.get_yaxis().set_visible(False)

    for ax in [ax10, ax11,ax12, ax14, ax15, ax16]:
        f.delaxes(ax)

    plt.tight_layout()
    plt.savefig('/home/thomas/pro/research/framework/res/articleII/loadings_articleII.png')

def predict(df_gcm, df_field_attributes, stamod, temporal_model_kind='dnn', obs=False):
    #     # chunks = np.linspace(0, len(df_field_attributes), len(df_field_attributes)/5)
    # for df_dem_chunk in np.array_split(df_field_attributes, 5):
    # chunck = df_field_attributes.iloc[int(init):int(end),:]

    # # For the article !!!!!
    # df_field_res= stamod.predict_model(chunk, df_gfs_by_hours,df_models_loadings, df_models_scores, model_scores_rf = False,
    #                            model_loading_curvfit=True, observed_scores = False, nb_PC =nb_pc)

    # df_field_res= stamod.predict_model(chunk, df_gcm, df_models_loadings, df_models_scores, dnn_model_scores= False,
    #                                    model_loading_curvfit=True, observed_scores = hourly_observed_scores, nb_PC =nb_pc)



    vars = []
    loads = []
    for model_name in stamod.keys():
        model = stamod[model_name]

        if obs:
            predicted =[]
            obs_scores = model['stamod'].scores
            # obs_scores = obs_scores.loc[(obs_scores.index.hour == 3) | (obs_scores.index.hour == 9) | (obs_scores.index.hour == 15) | (obs_scores.index.hour == 21),:]
            for df_chunk in np.array_split(obs_scores, 10):
                m = model['stamod_hourly'].predict_model(df_field_attributes, model_loadings=model['df_models_loadings'],model_loading_curvfit=True, obs_scores=df_chunk, nb_PC = model['stamod'].nb_PC)
                pred = m['predicted'].sum(axis=2)
                df_pred = pd.DataFrame(pred,index = m['index'])
                predicted.append(df_pred)
            predicted = pd.concat(predicted, axis=0)
            index = predicted.index
            predicted = predicted.values


        else:
            if temporal_model_kind == 'dnn':
                df_field_res = model['stamod'].predict_model(df_field_attributes, df_gcm, model['df_models_loadings'],
                                                  model['df_models_scores'][temporal_model_kind], model_loading_curvfit=True, nb_PC=model['stamod'].nb_PC,
                                                  dnn_model_scores=True)
                print('done pred dnn')
                #
                #
                # df_field_res = model['stamod'].predict_model(df_field_attributes, df_gcm, model['df_models_loadings'],
                #                                              model['df_models_scores'][temporal_model_kind],
                #                                              model_loading_curvfit=True, nb_PC=model['stamod'].nb_PC,
                #                                              dnn_model_scores=True)
            if temporal_model_kind == 'rnn':
                df_field_res = model['stamod'].predict_model(df_field_attributes, df_gcm, model['df_models_loadings'],
                                                             model['df_models_scores'][temporal_model_kind],
                                                             model_loading_curvfit=True, nb_PC=model['stamod'].nb_PC,
                                                             rnn_model_scores=True)
            if temporal_model_kind == 'lin':
                df_field_res = model['stamod'].predict_model(df_field_attributes, df_gcm, model['df_models_loadings'],
                                                             model['df_models_scores'][temporal_model_kind],
                                                             model_loading_curvfit=True, nb_PC=model['stamod'].nb_PC)


            predicted = df_field_res['predicted'].sum(axis=2)
            index = df_field_res['index']

        # total
        # df = pd.DataFrame(df_field_res['predicted'].sum(axis=2), index=df_gcm.index,
        #                   columns=df_dem_chunk.index).T
        #
        #

        # reshape
        shape_ribeirao = (239, 244) # Todo this must be generalized

        predicted = predicted.reshape(len(predicted), *shape_ribeirao)

        lat = df_field_attributes.loc[:,'lat'].values.reshape(shape_ribeirao)[:,0]
        lon = df_field_attributes.loc[:,'lon'].values.reshape(shape_ribeirao)[0,:]


        xr_var = xarray.DataArray(predicted, coords={'time':index, 'lat': lat, 'lon':lon}, dims=['time', 'lat', 'lon'])
        vars.append(xr_var)

        if not obs:
            loadings = df_field_res['loadings'].reshape(*shape_ribeirao, df_field_res['loadings'].shape[-1])
            xr_loadings = xarray.DataArray(loadings, coords={'lat': lat, 'lon':lon, 'pc_nb':range(1,df_field_res['loadings'].shape[-1]+1)},
                                           dims=['lat', 'lon','pc_nb'])
            loads.append(xr_loadings)

        # if not obs:
        # xr_var.coords['mask'] = (('lat', 'lon'), df_field_attributes['mask_posses'].values.reshape(shape_ribeirao))


    xrs = xarray.concat(vars, dim='var')
    xrs = xrs.assign_coords(var=list(stamod.keys()))
    xrs.attrs['units'] = str('T oC, Q g.kg-1, U m.s-1, V m.s-1, P Hpa')
    xrs.coords['mask'] = (('lat', 'lon'), df_field_attributes['mask_posses'].values.reshape(shape_ribeirao))

    if not obs:
        xrs_loadings = xarray.concat(loads, dim='var')
        xrs_loadings = xrs_loadings.assign_coords(var=list(stamod.keys()))
        xrs_loadings.coords['mask'] = (('lat', 'lon'), df_field_attributes['mask_posses'].values.reshape(shape_ribeirao))

        df_field_attributes = df_field_attributes.drop(['lat','lon'],axis=1)

        for col in df_field_attributes.columns:
            print(col)
            xrs_loadings.coords[col] = (('lat', 'lon'), df_field_attributes[col].values.reshape(shape_ribeirao))


    # df.to_csv(outpath + name + '_total_rec.csv', mode='a', header=None)

    # For the article
    # for i in range(df_field_res_t['predicted'].shape[0]):
    # for i in range(df_field_res['predicted'].shape[2]): # number of PCs
    #     pd.DataFrame(df_field_res['predicted'][:,:,i      ].T, columns = df_gfs_by_hours.index,
    #                  index= chunk.index).to_csv(outpath+name+'PC'+str(i+1)+'_rec.csv', header=None, mode='a')

    # # For the presentation
    # for i in range(df_field_res['predicted'].shape[2]): # number of PCs
    #     pd.DataFrame(df_field_res['predicted'][:,:,i].T, columns = hourly_observed_scores.index,
    #                  index= chunk.index).to_csv(outpath+name+'PC'+str(i+1)+'_rec.csv', header=None, mode='a')

    # Write loadigns
    # pd.DataFrame(df_field_res['loadings'][0,:,:]).to_csv(outpath+name+'PCloadings.csv', header=None, mode='a')


    if not obs:

        return xrs, xrs_loadings
    else:
        return xrs, None

class Framework(Framework):

    conf = Maihr_Conf()
    # notebook
    path = conf.framework_path

class predict_stamod_rib (luigi.Task, Framework):


    project_name = luigi.Parameter()
    version_name = luigi.Parameter(default='default')
    obs = luigi.Parameter(default=False)
    temporal_model_kind = luigi.Parameter(default='lin')

    From = luigi.Parameter()
    To = luigi.Parameter()

    decrease_res_by = luigi.Parameter(default=1)
    module_name = 'out'

    def output(self):
        output_fodler = self.path / self.project_name / self.module_name

        try:
            output_fodler.mkdir()
        except:
            pass


        if self.obs:
            filepath = str(output_fodler) + '/stamod_clim_obs.nc'
        else:
            From = pd.to_datetime(self.From)
            To = pd.to_datetime(self.To)
            filepath = str(output_fodler) + '/clim_dummy'+str(From)+"_"+str(To)+".luigi"

        return luigi.LocalTarget(str(filepath))



    def run(self):


        stamod = train_stamod(self.path, remove_mean=False)

        print('finish training')
        From = pd.to_datetime(str(self.From))
        To = pd.to_datetime(str(self.To))

        # Get field dataframe
        field_folder = self.path / 'database/predictors/out/df_field/ribeirao/'
        df_field_topex_alt = pd.read_csv(str(field_folder / 'df_field_ribeirao.csv'), index_col=0)
        df_field_dist_outlet = pd.read_csv(str(field_folder / "df_field_ribeirao_dist_outlet.csv"), index_col=0)
        df_field_dist_river = pd.read_csv(str(field_folder / "df_field_ribeirao_dist_river.csv"), index_col=0)
        df_field_dist_side = pd.read_csv(str(field_folder / "df_field_ribeirao_dist_side.csv"), index_col=0)

        df_field_attributes = pd.concat([df_field_topex_alt, df_field_dist_outlet, df_field_dist_river, df_field_dist_side], axis=1,join='inner')

        # # For the presentation
        # hourly_observed_scores_T = stamod_hourly_T.scores.groupby(lambda x:x.hour).mean()
        # hourly_observed_scores_T.iloc[:,[2,3]] = hourly_observed_scores_T.iloc[:,[2,3]]*-1 # PC2, PC3
        #
        # hourly_observed_scores_Q = stamod_hourly_Q.scores.groupby(lambda x:x.hour).mean()
        # hourly_observed_scores_Q.iloc[:,1] = hourly_observed_scores_Q.iloc[:,1]*-1 # PC2, PC3
        #
        # hourly_observed_scores_U = stamod_hourly_U.scores.groupby(lambda x:x.hour).mean()
        # hourly_observed_scores_U.iloc[:,0] = hourly_observed_scores_U.iloc[:,0]*-1 # PC2, PC3
        #
        # hourly_observed_scores_V = stamod_hourly_V.scores.groupby(lambda x:x.hour).mean()
        # hourly_observed_scores_V.iloc[:,0] = hourly_observed_scores_V.iloc[:,0]*-1 # PC2, PC3

        # outpath = "/home/thomas/pro/research/framework/model/out/local_downscaling/ribeirao_articleII/chunks/"

        # for model in self.input().values:

        df_gcm = pd.read_csv(str(self.path / 'database/predictors/out/gcm_data/local/ecmwf/df_era_interim_historical.csv'), index_col=0, parse_dates=True)


        years = range(From.year,To.year+1) # need + 1 to include the year

        for year in years:
            print(year)
            start_year = pd.to_datetime(str(year))
            end_year = start_year + offsets.YearEnd()

            print(start_year)
            print(end_year)
            print(df_gcm)
            df_gcm_select = df_gcm.loc[start_year:end_year, :]
            df_gcm_select.index = df_gcm_select.index - datetime.timedelta(hours=3)
            df_gcm_select.dropna(axis=1, how='all', inplace=True)
            df_gcm_select.dropna(axis=0, how='any', inplace=True)

            print('strat predict article II')
            xr, xr_loadings = predict(df_gcm_select, df_field_attributes, stamod, temporal_model_kind = self.temporal_model_kind, obs=self.obs)

            print('finish predict')

            print('get new lat')
            # resolution
            new_lat = np.linspace(xr['lat'].min(), xr['lat'].max(), len(xr['lat']) / self.decrease_res_by)
            new_lon = np.linspace(xr['lon'].min(), xr['lon'].max(), len(xr['lon']) / self.decrease_res_by)

            print('interp')
            xr = xr.interp(lat=new_lat, lon=new_lon)

            xr.attrs['Description'] = str('stamod lcb ribeirao' + '  version - ' + self.version_name)

            if not self.obs:
                ds_loadings = xr_loadings.to_dataset(name='loadings')
                ds_loadings.to_netcdf(str(self.path / self.project_name / self.module_name/'loadings.nc'))

            print('to netcdf')
            ds = xr.to_dataset(name='clim')
            print(ds)
            file_path = os.path.dirname(self.output().path)+'/clim_'+str(start_year)+'.nc'
            print(file_path)
            ds.to_netcdf(file_path)

        open(self.output().path, 'a').close()

            # write_reconstruct(df_gcm, df_field_attributes, model_T, outpath,temporal_model_name, name="T", nb_pc=4)
        #     write_reconstruct(df_gcm, df_field_attributes, model_Q, outpath, name="Q", nb_pc=2)
        #     write_reconstruct(df_gcm, df_field_attributes, model_U, outpath, name="U", nb_pc=2)
        #     write_reconstruct(df_gcm, df_field_attributes, model_V, outpath, name="V", nb_pc=3)
        #     write_reconstruct(df_gcm, df_field_attributes, model_P, outpath, name="P", nb_pc=2)

def plot_validation_individual_station(lat, lon, name, varname, df_obs, xr_mod, df_rain, wind_ridge, csf, axe, color):

    # for axe,hour in zip (axes, hours):
    # x = xr_obs.sel(var=varname).sel(lat=lat, lon=lon, method='nearest').where(
    #     xr_bias_lin.time.dt.hour == hour).to_array()


    df_pred = xr_mod.sel(var=varname).sel(lat=lat, lon=lon, method='nearest').to_dataframe()
    df_pred = df_pred.loc[:,'clim']
    df_pred.dropna(axis=0, how='any', inplace=True)

    idx = common_index(df_obs.index, df_pred.index)
    y = df_pred.loc[idx]
    x = df_obs.loc[idx]
    #
    xmin = np.min(x)
    xmax = np.max(x)

    axe.scatter(x=x, y=y, s=10,c=color, alpha=0.5)

    idx_rain = df_rain[(df_rain > 10)].index
    x_rain = df_obs.loc[idx_rain]
    y_rain =df_pred.loc[idx_rain]
    axe.scatter(x=x_rain, y=y_rain, c='r', s=20)
    #
    wind_ridge = wind_ridge[df_pred.index]
    idx_wind = wind_ridge[wind_ridge > 15].index
    x_wind = df_obs.loc[idx_wind]
    y_wind =df_pred.loc[idx_wind]
    axe.scatter(x=x_wind, y=y_wind, c='b', s=20)

    wind_ridge = wind_ridge[df_pred.index]
    csf = csf[df_pred.index]
    idx_highcsf = csf[csf > 1.2].index
    idx_wind = wind_ridge[(wind_ridge < 1)].index
    x_wind = df_obs.loc[idx_wind]
    y_wind =df_pred.loc[idx_wind]
    axe.scatter(x=x_wind, y=y_wind, c='purple', s=20)



    # x_highcsf = df_obs[idx_csf]
    # y_highcsf = df_pred.loc[idx_csf]
    # axe.scatter(x=x_highcsf, y=y_highcsf, c='g', s=20)
    # #
    # csf = csf[df_pred.index]
    # idx_csf = csf[csf < 0.1].index
    # x = df_obs[idx_csf]
    # y = df_pred.loc[idx_csf]
    # axe.scatter(x=x, y=y, c='y', s=20)

    axe.set_xlabel('Observed')
    axe.set_title(varname + ' at' + name)
    axe.plot(np.arange(xmin, xmax), np.arange(xmin, xmax), c='k')

    return axe

def plot_scatter_obs(xr_mod):

    lat_vale = -22.88097
    lon_vale = -46.249083
    name_vale = ''

    lat_topo = -22.87686
    lon_topo = -46.254528
    name_topo = ''

    df_T = pd.read_csv('/home/thomas/pro/research/framework/database/stations_database/4_fill/out/articleII/df_Ta.csv',index_col=0, parse_dates=True)
    df_T.dropna(axis=0, how='any', inplace=True)

    df_Q = pd.read_csv('/home/thomas/pro/research/framework/database/stations_database/4_fill/out/articleII/df_Qa.csv',index_col=0, parse_dates=True)
    df_Q.dropna(axis=0, how='any', inplace=True)

    df_U = pd.read_csv('/home/thomas/pro/research/framework/database/stations_database/4_fill/out/articleII/df_Uw.csv',index_col=0, parse_dates=True)
    df_U.dropna(axis=0, how='any', inplace=True)

    df_V = pd.read_csv('/home/thomas/pro/research/framework/database/stations_database/4_fill/out/articleII/df_Vw.csv',index_col=0, parse_dates=True)
    df_V.dropna(axis=0, how='any', inplace=True)

    df_rain = pd.read_csv('/home/thomas/pro/research/framework/database/stations_database/2_dataframe/out/articleII_prec/df_Pr.csv',index_col=0, parse_dates=True)  # Their is no C09 inthe fill ...
    df_rain = df_rain.loc[pd.to_datetime(xr_mod.time.values)].mean(axis=1)

    df = pd.read_csv('/home/thomas/pro/research/framework/database/stations_database/2_dataframe/out/articleII/df_Sw.csv',index_col=0, parse_dates=True)  # Their is no C09 inthe fill ...
    wind_ridge = df.loc[:, 'rib_C09']

    csf = get_csf()

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(14, 3))

    plot_validation_individual_station(lat_topo, lon_topo, name_topo, 'T', df_T.loc[:, 'rib_C07'], xr_mod,df_rain, wind_ridge, csf, ax1, color='k')
    plot_validation_individual_station(lat_topo, lon_topo, name_topo, 'Q', df_Q.loc[:, 'rib_C07'], xr_mod,df_rain, wind_ridge, csf, ax2, color='k')
    plot_validation_individual_station(lat_topo, lon_topo, name_topo, 'V', df_V.loc[:, 'rib_C07'], xr_mod,df_rain, wind_ridge, csf, ax3, color='k')
    plot_validation_individual_station(lat_topo, lon_topo, name_topo, 'U', df_U.loc[:, 'rib_C07'], xr_mod,df_rain, wind_ridge, csf, ax4, color='k')


    plot_validation_individual_station(lat_vale, lon_vale, name_vale, 'T', df_T.loc[:, 'rib_C04'], xr_mod,df_rain, wind_ridge, csf, ax1, color='0.5')
    plot_validation_individual_station(lat_vale, lon_vale, name_vale, 'Q', df_Q.loc[:, 'rib_C04'], xr_mod,df_rain, wind_ridge, csf, ax2, color='0.5')
    plot_validation_individual_station(lat_vale, lon_vale, name_vale, 'V', df_V.loc[:, 'rib_C04'], xr_mod,df_rain, wind_ridge, csf, ax3, color='0.5')
    plot_validation_individual_station(lat_vale, lon_vale, name_vale, 'U', df_U.loc[:, 'rib_C04'], xr_mod,df_rain, wind_ridge, csf, ax4, color='0.5')





    ax1.set_ylabel('Estimated')
    ax1.set_xlabel('Observed')
    ax2.set_xlabel('Observed')
    ax3.set_xlabel('Observed')
    ax4.set_xlabel('Observed')
    # ax5.set_ylabel('Valley station \n Estimated')

    axs = [ax1, ax2, ax3, ax4]
    for t, ax in zip(["a)", "b)", 'c)', 'd)', 'e)', 'f)', 'g)', 'h)'], axs):
        ax.text(-0.05, 1, t, transform=ax.transAxes, size=14, weight='bold')

    for ax in axs:
        ax.set_title('')

    ax1.set_title(r'Temperature ($^\circ C$)', size=10)
    ax2.set_title(r'Specific humidity ($g.kg^{-1}$)', size=10)
    ax3.set_title(r'Meridional wind ($m.s^{-1}$)', size=10)
    ax4.set_title(r'Zonal wind ($m.s^{-1}$)', size=10)

    # plt.tight_layout()
    plt.savefig('/home/thomas/pro/research/framework/res/articleII/comparison_with_stations.png')

def train_stamod(path,remove_mean = False):
    From = '2015-03-01 00:00:00'
    To = '2016-04-01 00:00:00'

    path = Path(path)

    # ===============================================================================
    # Get scores predictors ERA interim
    # ===============================================================================
    df_gcm = pd.read_csv(str(path / 'database/predictors/out/gcm_data/local/ecmwf/df_era_interim_historical.csv'), index_col=0,parse_dates=True)

    df_gcm = df_gcm.loc[From:To,:]
    df_gcm.index = df_gcm.index - datetime.timedelta(hours=3)
    df_gcm.dropna(axis=1,how='all', inplace=True)
    df_gcm.dropna(axis=0,how='any', inplace=True)

    # pipeline = Pipeline([('quantile_transform', QuantileTransformer()),('scaling', StandardScaler())])
    # df_gcm = pd.DataFrame(pipeline.fit_transform(df_gcm.values), index=df_gcm.index, columns=df_gcm.columns)

    # ===============================================================================
    # Create surface observations dataframe
    # ===============================================================================
    data_folder_path = path / 'database/stations_database/4_fill/out/articleII/'

    data_folder_path_rain = str(path / 'database/stations_database/4_fill/out/articleII_prec/hourly/')
    AttSta = get_attributes(path)
    models_results = {}

    #==============================================
    # zonal wind

    path_data = str(data_folder_path / 'df_Uw.csv')


    if remove_mean:
        nb_pc = 2
        staname_to_drop = ['rib_C19','rib_C09','rib_C15']
        pred_load_names = ['dist_side', 'dist_side','Alt']
        manual_selection = {
            1: ['skt_var_dt', 'v10', 'z_var_850dt'],
            2: ['t_var_2850', 'z_var_850dx', 'v_var_850', 'u10']}
        fits_loadings = [lin, lin,lin]

    else:
        nb_pc = 2
        staname_to_drop = ['rib_C13', 'rib_C19', 'rib_C09']
        pred_load_names = ['dist_side', 'topex_w','Alt']
        manual_selection = {
            1: ['skt_var_dt', 'v10', 'z_var_850dt'],
            2: ['t_var_2850', 'z_var_850dx', 'v_var_850', 'u10']}
        fits_loadings = [lin, lin,lin]
    model_U = modeling(From, To, path_data, df_gcm, AttSta, nb_pc, pred_load_names, fits_loadings, staname_to_drop,
                       manual_selection=manual_selection, var='U',remove_mean=remove_mean, framework_path=str(path))


    #==============================================
    # # Temperature
    path_data = str(data_folder_path / 'df_Ta.csv')
    nb_pc = 4

    if remove_mean:
        staname_to_drop = ['rib_C11']

        pred_load_names = [['Alt', 'dist_outlet'], ['Alt', 'dist_outlet'],['dist_outlet', 'Alt'], 'dist_river']
        fits_loadings = [multi_pol2_lin , multi_pol2_lin, lin2, lin]
        # pred_load_names = ['Alt', 'Alt', 'dist_outlet', 'dist_side' ]# TOdo warning: just to make the loadings plot
        # fits_loadings = [pol2, pol2, lin, lin]
        manual_selection = {
            1: ['t2m', 'skt_var_dt', 't_var_2850'],
            2: ['t_var_2900', 't_var_850', 'skt','r_var_900'],
            3: ['skt_var_dt', 'v_var_10900'],
            4: ['msl_var_dt', 'slhf', 't_var_850900']}
    else:

        # staname_to_drop = ['rib_C19','rib_C11']
        staname_to_drop = ['rib_C11']
        pred_load_names = [['Alt', 'dist_outlet'], ['Alt', 'dist_outlet'],['dist_side', 'Alt'], 'dist_river']
        fits_loadings = [multi_pol2_lin , multi_pol2_lin, lin2, piecewise_linear_pc3_ribeirao]
        #
        # pred_load_names = ['Alt', 'Alt', 'dist_outlet', 'dist_side' ]# TOdo warning: just to make the loadings plot
        # fits_loadings = [pol2, pol2, lin, lin ]

        manual_selection = {
            1: ['t2m', 'skt_var_dt', 't_var_2850'],
            2: ['t_var_2900', 't_var_850', 'skt','r_var_900'],
            3: ['skt_var_dt', 'v_var_10900'],
            4: ['msl_var_dt', 'slhf', 't_var_850900']}

    model_T = modeling(From, To, path_data, df_gcm, AttSta, nb_pc, pred_load_names, fits_loadings, staname_to_drop,
                       manual_selection=manual_selection,var='T',remove_mean=remove_mean, framework_path=str(path))



    # ==============================================
    path_data = str(data_folder_path / 'df_Vw.csv')

    nb_pc = 2

    if remove_mean:
        staname_to_drop = ['rib_C09', 'rib_C16']
        pred_load_names = ['topex_ne', ['dist_river','dist_side']]

        pred_load_names = ['topex_ne', 'dist_river']

        manual_selection = {
            1: ['v_var_900', 'str', 't_var_850dy'],
            2: ['u_var_850', 'd2m_var_dt']}
        fits_loadings = [lin, lin, lin]
    else:
        staname_to_drop = ['rib_C09', 'rib_C19']
        pred_load_names = ['topex_ne', 'dist_side']
        manual_selection = {
            1: ['v_var_900', 'str', 't_var_850dy'],
            2: ['u_var_850']}
        fits_loadings = [lin, lin, lin]

    model_V = modeling(From, To, path_data, df_gcm, AttSta, nb_pc, pred_load_names, fits_loadings, staname_to_drop,
                       manual_selection=manual_selection, var='V',
                       remove_mean=remove_mean, framework_path=str(path))  # Todo Make the manual selection of the Era Interim


    #==============================================
    # Specific humidity
    path_data = str(data_folder_path / 'df_Qa.csv')
    nb_pc = 2
    if remove_mean:
        pred_load_names = [['dist_river','dist_side'], 'Alt']
        fits_loadings = [lin2, pol2]

        # pred_load_names = ['dist_river', 'Alt']
        # fits_loadings = [lin, pol2] # TOdo warning: just to make the loadings plot

        manual_selection = {
            1: ['d2m', 'skt', 'q_var_850'],
            2: ['p83.162', 'r_var_900']}
    else:
        staname_to_drop = ['rib_C13']  # Todo ok !

        pred_load_names = [['Alt','dist_side'], 'Alt']
        fits_loadings = [multi_pol2_lin, pol2]

        # pred_load_names = ['Alt', 'Alt']
        # fits_loadings = [lin, pol2] # TOdo warning: just to make the loadings plot

        manual_selection = {
            1: ['d2m', 'skt', 'q_var_850'],
            2: ['p83.162', 'r_var_900', 'p85.162_var_dt','d2m_var_dt']}

    model_Q = modeling(From, To, path_data, df_gcm, AttSta, nb_pc, pred_load_names, fits_loadings, staname_to_drop,
                       manual_selection=manual_selection, var='Q',remove_mean=remove_mean, framework_path=str(path))



    # Pressure
    path_data = str(data_folder_path / 'df_Pa.csv')
    staname_to_drop = ['rib_C08']
    nb_pc = 2
    pred_load_names = ['Alt', 'Alt']
    fits_loadings = [lin, lin]
    model_P = modeling(From, To, path_data, df_gcm, AttSta, nb_pc, pred_load_names, fits_loadings, staname_to_drop,
                       manual_selection=None, var='P', framework_path=str(path))  # Todo Make the manual selection of the Era Interim

    print('finish modeling')
    stamod = {'T': model_T, 'Q': model_Q,
              'U': model_U, 'V': model_V,'P':model_P}
    return stamod # Todo Warning cant be pickled/joblibed

def modeling(From, To, path_data, df_gcm, AttSta, nb_pc, pred_load_names, fits_loadings, staname_to_drop, manual_selection, var, remove_mean=False, framework_path=False):

    print('X'*80)
    print("MODELING "+ str(var))
    print('X'*80)

    df = pd.read_csv(path_data, index_col=0, parse_dates=True)
    df.drop(staname_to_drop, axis=1, inplace=True)
    df.dropna(axis=0,how='any', inplace=True)

    df = df.loc[From:To, :]

    if remove_mean:
        if var =='T':
            mean_data = df_gcm.loc[:,'t2m'] - 273.15
            df = df.sub(mean_data, axis=0)
            df.dropna(axis=0,inplace=True)
        if var =='Q':
            mean_data = df_gcm.loc[:,'q2m']
            df = df.sub(mean_data, axis=0)
            df.dropna(axis=0,inplace=True)
        if var =='U':
            mean_data = df_gcm.loc[:,'u10']
            df = df.sub(mean_data, axis=0)
            df.dropna(axis=0,inplace=True)
        if var =='V':
            mean_data = df_gcm.loc[:,'v10']
            df = df.sub(mean_data, axis=0)
            df.dropna(axis=0,inplace=True)


    df_gcm_pca = perform_pca_byvar(df_gcm) # Todo: make PCA by var of all levels for Era interim
    train_index, test_index, full_index = get_train_test_index_columns(df, df_gcm)
    df_attributes = AttSta.attributes[AttSta.attributes.index.isin(df.columns)]

    df = df.loc[:,df.columns.isin(df_attributes.index)]
    df_attributes = df_attributes.loc[df.columns,:]

    df_train = df.loc[train_index,:]
    df_gfs_train = df_gcm.loc[train_index, :]
    df_gfs_pca_train = df_gcm_pca.loc[train_index,:]

    df_test = df.loc[test_index,:]
    df_gfs_test = df_gcm.loc[test_index, :]
    df_gfs_pca_test = df_gcm_pca.loc[test_index, :]


    df_full = df.loc[full_index,:]
    df_full.dropna(axis=0,how='any', inplace=True)

    #pca
    stamod_hourly = StaMod(df, AttSta)
    stamod_hourly.pca_transform(nb_PC=nb_pc, cov=True, standard=False, remove_mean1 =False, remove_mean0 =False)


    stamod = StaMod(df_full, AttSta)
    stamod.pca_transform(nb_PC=nb_pc, cov=True, standard=False, remove_mean1 =False, remove_mean0 =False)

    #=========================================================================
    # Modeling Loadings
    #=========================================================================

    predictors_loadings = []
    for pred_load_name in pred_load_names:
        predictors_loadings.append(df_attributes[pred_load_name])

    # Fit loadings
    loadings = pd.DataFrame(stamod.eigenvectors.T)
    df_models_loadings =  stamod.fit_curvfit(predictors_loadings,loadings, fits=fits_loadings)

    #=========================================================================
    # Modeling Scores
    #=========================================================================
    df_models_scores={}
    res = {}

    # train_scores = stamod.scores.loc[train_index,:]
    # pipeline = Pipeline([('quantile_transform', QuantileTransformer()),('scaling', StandardScaler())])
    # train_scores = pd.DataFrame(pipeline.fit_transform(train_scores.values), index=train_scores.index, columns=train_scores.columns)


    #===================================================================================================================================================
    # Linear
    if manual_selection == None:
        df_models_scores['lin'], log_lin = stamod.stepwise_model(stamod.scores.loc[train_index,:], df_gfs_train, lim_nb_predictors=4,
                                                        constant=True, log=True, manual_selection=None) # stepwise multi-linear regression
        # lin - quantile mapping
        df_models_scores['lin-quantile-full'], log = stamod.stepwise_model(stamod.scores.loc[full_index, :],df_gcm.loc[full_index, :],
                                                                           lim_nb_predictors=4, constant=True,log=True,manual_selection=None)

    else:
        df_models_scores['lin'], log_lin = stamod.stepwise_model(stamod.scores.loc[train_index,:], df_gfs_train,lim_nb_predictors=None, constant=True, log=True, manual_selection=manual_selection)
        df_models_scores['lin-quantile-full'], log = stamod.stepwise_model(stamod.scores.loc[full_index, :],df_gcm.loc[full_index, :],
                                                                           lim_nb_predictors=None, constant=True,log=True,manual_selection=manual_selection)



    res['lin'] = stamod.predict_model(df_attributes, df_gfs_test,df_models_loadings,df_models_scores['lin'], model_loading_curvfit=True, nb_PC =nb_pc)
    res['lin-q'] = stamod.predict_model(df_attributes, df_gcm, df_models_loadings,df_models_scores['lin-quantile-full']
                                        , model_loading_curvfit=True, nb_PC =nb_pc, idx_test=df_gfs_test.index, idx_train=df_gfs_train.index, eqm=True)

    #
    # # # #===================================================================================================================================================
    # # # #DNN



    model_path = str(framework_path)+ "/model/out/nn/" +var+'.h5'
    df_models_scores['dnn']= stamod.dnn_model(df_gfs_pca_train, stamod.scores.loc[train_index,:], epochs=150, model_path=model_path)
    print('done train dnn')
    res['dnn'] = stamod.predict_model(df_attributes, df_gfs_pca_test,df_models_loadings,df_models_scores['dnn'], model_loading_curvfit=True, nb_PC =nb_pc, dnn_model_scores=True)
    print('done pred dnn')

    # # # #===================================================================================================================================================
    # # # DNN QUANTILE
    # model_path = "/home/thomas/pro/research/framework/model/out/nn/" + var + 'quantile.h5'
    # df_models_scores['dnn-full-quantile']= stamod.dnn_model(df_gcm.loc[full_index, :], stamod.scores.loc[full_index, :], epochs=100, model_path=model_path)
    # # df_models_scores['dnn-train-quantile']= stamod.dnn_model(df_gfs_pca_train, stamod.scores.loc[train_index,:])
    # res['dnn-full-quantile'] = stamod.predict_model(df_attributes, df_gcm.loc[full_index, :],df_models_loadings,df_models_scores['dnn-full-quantile'],
    #                                                 model_loading_curvfit=True, nb_PC =nb_pc, dnn_model_scores=True, qm_maps=None)
    #
    # # quantile mapping
    # scores_obs = stamod.scores
    # scores_pred = pd.DataFrame(res['dnn-full-quantile']['scores'][0,:,:], columns=scores_obs.columns, index = scores_obs.index)
    #
    # qm_maps = {}
    # for pc_nb in range(scores_obs.shape[1]):
    #     qmap = qm(scores_obs.loc[:,pc_nb+1], scores_pred.loc[:,pc_nb+1],10)
    #     qm_maps[pc_nb + 1] = qmap
    #
    # res['dnn-q'] = stamod.predict_model(df_attributes, df_gfs_pca_test,df_models_loadings,df_models_scores['dnn'], model_loading_curvfit=True, nb_PC =nb_pc, dnn_model_scores=True, qm_maps=qm_maps)
    #
    # del res['dnn-full-quantile']
    # del res['dnn-test-quantile']
    # del res['lin-quantile-train']
    # del res['lin-quantile-test']
    #
    # #
    #
    # # # # RNN
    # model_path = "/home/thomas/pro/research/framework/model/out/nn/" + var + 'rnn.h5'
    #
    # df_models_scores['rnn']= stamod.rnn_model(df_gfs_pca_train, stamod.scores.loc[train_index,:], SEQUENCE_LENGTH=4, epochs=2, model_path=model_path)
    # res['rnn'] = stamod.predict_model(df_attributes, df_gfs_pca_test, df_models_loadings, df_models_scores['rnn'], model_loading_curvfit=True, nb_PC =nb_pc, rnn_model_scores=True)


    #=========================================================================
    # validation
    #=========================================================================

    verif = defaultdict(dict)


    for model in res.keys():

        if remove_mean:
            # # REMOVING MEAN===============================
            verif['bias'][model]=stamod.skill_model(df_test, res[model], metrics = None, summary=False, use_bias=True, plot_summary=False,mean_data=mean_data)
            verif['R'][model]=stamod.skill_model(df_test, res[model], metrics = pearsonr, summary=False, plot_summary=False,mean_data=mean_data)
            verif['R2'][model]=stamod.skill_model(df_test, res[model], metrics = metrics.r2_score, summary=False, plot_summary=False,mean_data=mean_data)
            verif['MAE'][model] =  stamod.skill_model(df_test,res[model], metrics = metrics.mean_absolute_error, summary=False, plot_summary=False,mean_data=mean_data)
            verif['MSE'][model] = stamod.skill_model(df_test,res[model], metrics = metrics.mean_squared_error, summary=False, plot_summary=False,mean_data=mean_data)
            verif['RMSE'][model]=verif['MSE'][model].pow(1./2)
            verif['exp_var'][model]= stamod.skill_model(df_test, res[model], metrics = metrics.explained_variance_score, summary=False, plot_summary=False,mean_data=mean_data)
        else:
            # # REMOVING MEAN===============================
            verif['bias'][model]=stamod.skill_model(df_test, res[model], metrics = None, summary=False, use_bias=True, plot_summary=False,mean_data=None)
            verif['R'][model]=stamod.skill_model(df_test, res[model], metrics = pearsonr, summary=False, plot_summary=False,mean_data=None)
            verif['R2'][model]=stamod.skill_model(df_test, res[model], metrics = metrics.r2_score, summary=False, plot_summary=False,mean_data=None)
            verif['MAE'][model] =  stamod.skill_model(df_test,res[model], metrics = metrics.mean_absolute_error, summary=False, plot_summary=False,mean_data=None)
            verif['MSE'][model] = stamod.skill_model(df_test,res[model], metrics = metrics.mean_squared_error, summary=False, plot_summary=False,mean_data=None)
            verif['RMSE'][model]=verif['MSE'][model].pow(1./2)
            verif['exp_var'][model]= stamod.skill_model(df_test, res[model], metrics = metrics.explained_variance_score, summary=False, plot_summary=False,mean_data=None)


    scores_r2 = {}

    for model in res.keys():
        scores_r2[model] = stamod.skill_scores(stamod.scores.loc[res[model]['index'],:], res[model]['scores'][0,:,:], metrics = metrics.r2_score)

    model = {'stamod':stamod, 'stamod_hourly':stamod_hourly, 'df_models_loadings':df_models_loadings,
             'df_models_scores':df_models_scores, 'verif': verif, 'res':res, 'scores_r2': scores_r2,'log':log_lin}

    # K.clear_session()
    # gc.collect()

    return model

def predict_csf_posses(model_path):
    """
    Example
    model_path = "/home/thomas/pro/research/framework/model/out/statmod/irr_rib_model/csf_model_v8.h5"
    pred_csf = predict_csf_posses(model_path)



    :return: dataframe, predicted clear sky factor
    """


    conf = Maihr_Conf()
    database = conf.framework_path / 'database'

    # lat_rib = -22.88
    # lon_rib = 360 - 46.24

    datafolder_path = '/home/thomas/pro/research/framework/gcm_database/ecmwf/data/'
    df_gcm = pd.read_csv(str(Path(database) / 'predictors/out/gcm_data/local/ecmwf/df_era_interim_historical.csv'), index_col=0,parse_dates=True)
    df_gcm.dropna(inplace=True,axis=0, how='any')

    xr_surf_for = xarray.open_dataset(datafolder_path + 'ecmwf-22.8_313.5_-23_314.25_surface_forecast.nc')
    # # GCM solar radiation
    gcm_solar_radiation = xr_surf_for[['ssrd', 'ssr', 'ssrc', 'tisr']]

    gcm_csf = gcm_solar_radiation['ssrd']/gcm_solar_radiation['tisr']
    gcm_csf = gcm_csf.interp(latitude=lat_rib, longitude=lon_rib)
    gcm_csf.name ='gcm_csf'
    df_gcm_csf = gcm_csf.to_dataframe()['gcm_csf']
    df_gcm_csf.dropna(inplace=True)


    if not os.path.exists(model_path):
        obs_csf = get_csf()
        obs_csf.dropna(inplace=True, axis=0, how='any')
        #
        # csf = pd.concat([df_gcm_csf, obs_csf],axis=1, join='inner')

        # Model
        idx = common_index(df_gcm.index, obs_csf.index)
        df_gcm = df_gcm.loc[idx, :]
        obs_csf = obs_csf.loc[idx]

        scaler_gcm = MinMaxScaler(feature_range=(0, 1))
        scaler_obs_csf = MinMaxScaler(feature_range=(0, 1))

        scaler_gcm = scaler_gcm.fit(df_gcm.values)
        scaler_obs_csf = scaler_obs_csf.fit(obs_csf.values[:, np.newaxis])

        joblib.dump(scaler_gcm, os.path.dirname(model_path)+'scaler_gcm.pickle')
        joblib.dump(scaler_obs_csf, os.path.dirname(model_path)+'scaler_obs.pickle')

        df_gcm_scaled = pd.DataFrame(scaler_gcm.transform(df_gcm.values), index=df_gcm.index, columns=df_gcm.columns)
        obs_csf_scaled = pd.DataFrame(scaler_obs_csf.transform(obs_csf.values[:, np.newaxis]), index=obs_csf.index,columns=['csf'])

        X_train, X_test, y_train, y_test = train_test_split(df_gcm_scaled.values, obs_csf_scaled, test_size=0.20, random_state=1)
        model, history = DNN_ribeirao(X_train, y_train, epochs=100, model_path=model_path, plot=False)

        # csf_pred = pd.DataFrame(model.predict([df_gcm_scaled.values]), index=df_gcm_scaled.index, columns=['pred'])
        #
        # # csf = pd.concat([csf,y_pred],axis=1)
        # obs_csf = pd.concat([obs_csf, csf_pred], axis=1)
        #
        # # ax = csf.plot(kind='scatter',x='csf_obs',y='gcm_csf',c='b', alpha=0.5, s=30,label='GCM CSF')
        # ax = obs_csf.plot(kind='scatter', x='csf', y='pred', c='k', alpha=0.5, s=30, label='Downscaled CSF')
        # plt.legend()

        save_model(model, model_path)
    else:
        model = load_model(model_path)
        scaler_gcm = joblib.load(os.path.dirname(model_path) + 'scaler_gcm.pickle')
        scaler_obs_csf =joblib.load(os.path.dirname(model_path) + 'scaler_obs.pickle')

    csf_pred = model.predict([scaler_gcm.transform(df_gcm.values)])
    csf_pred = pd.DataFrame(scaler_obs_csf.inverse_transform(csf_pred), index=df_gcm.index, columns=['csf_pred'])

    return csf_pred


if __name__=='__main__':
    global framework_path
    conf = Maihr_Conf()

    # notebook
    framework_path = conf.framework_path

    # # article
    project_name = 'stamod_rib'
    version_name = 'articleII'
    # #
    # luigi.build([predict_stamod_rib(project_name=project_name, version_name=versiI have to bs=False)], local_scheduler=True)
    #
    # luigi.build([predict_stamod_rib(project_name=project_name, version_name=version_name,From='2015-03-01 00:00:00', To='2016-04-01 00:00:00',
    #                                 obs=False, decrease_res_by=1)], local_scheduler=True)


    path = Path(Framework.path)
    stamod = train_stamod(path, remove_mean=False)
    # # # # #
    # # # # # # # ===============================================================================
    # # # # # # # Get scores predictors ERA interim
    # # # # # # # ===============================================================================
    df_gcm = pd.read_csv(str('/vol0/thomas.martin/test/df_era_interim.csv'), index_col=0,parse_dates=True)

    df_gcm.index = df_gcm.index - datetime.timedelta(hours=3)
    df_gcm.dropna(axis=1,how='all', inplace=True)
    df_gcm.dropna(axis=0,how='any', inplace=True)

    df_gcm_t = df_gcm.loc[:,'t2m'] - 273.15
    df_gcm_u = df_gcm.loc[:,'u10']
    df_gcm_v = df_gcm.loc[:,'v10']
    df_gcm_q = df_gcm.loc[:,'q2m']
    #
    # print('done')
    # # #
    # # # # ===============================================================================
    # # # # Plots
    # # # # ===============================================================================
    # axs = plot_loadings(stamod['T'], stamod['Q'], stamod['U'], stamod['V']) # Todo ok but I need to use only one spatial predictor for the temperature !
    # plot_scores_articleII(stamod['T']['stamod'], stamod['Q']['stamod'], stamod['U']['stamod'], stamod['V']['stamod']) # Todo ok!
    # plot_daily_scores_articleII(stamod['T']['stamod_hourly'], stamod['Q']['stamod_hourly'], stamod['U']['stamod_hourly'], stamod['V']['stamod_hourly'],
    #                             stamod['T'], stamod['Q'],stamod['U'], stamod['V']) # Todo ok!

    # #
    # # # # # print_explained_var(stamod['T'], stamod['Q'], stamod['U'], stamod['V']) # Todo ok !
    #
    #
    # plot_rmse_by_quantile(stamod['T'],stamod['Q'],stamod['U'], stamod['V'],df_gcm_t, df_gcm_q, df_gcm_u, df_gcm_v,remove_mean=False) # Todo ok!

    print_skill_articleII(stamod, df_gcm_t,df_gcm_q,df_gcm_u,df_gcm_v) # Todo ok!
    # quantiles_plot(stamod['T'], stamod['Q'], stamod['U'], stamod['V'],df_gcm_t, df_gcm_q, df_gcm_u, df_gcm_v,remove_mean=False) # Todo Ok !
    #
    #


    #
    # xr_obs = xarray.open_dataset('/home/thomas/pro/research/framework/stamod_rib/out/stamod_clim_obs.nc')
    xr_loadings = xarray.open_dataset('/home/thomas/pro/research/framework/stamod_rib/out/loadings.nc')
    # # xr_mod_dnn = xarray.open_dataset('/home/thomas/pro/research/framework/stamod_rib/out/stamod_climdnn_2015_2016.nc')
    # xr_mod_lin = xarray.open_dataset('/home/thomas/pro/research/framework/stamod_rib/out/stamod_climlin_2015_2016.nc')
    # # xr_mod_rnn = xarray.open_dataset('/home/thomas/pro/research/framework/stamod_rib/out/stamod_climrnn_2015_2016.nc')
    # # res_sib = xarray.open_dataset('/home/thomas/database_outdropbox/sib2_ribdata1_pastagem.nc')
    # xr_bias_lin = xr_mod_lin - xr_obs
    #
    # plot_field_var(xr_mod_lin, filename='field_reconstruct.png') # Todo ok ! # Todo for the validation set
    #
    # plot_field_var(xr_bias_lin, filename='spatial_bias.png') # Todo ok ! # Todo for the validation set
    # plot_fields_spatial_predictors(xr_loadings) # Todo ok!

    # plot_field_loadings(xr_loadings) # Todo ok!

    # plot_scatter_obs(xr_mod_lin) # Todo ok!
    # plot_scatter_obs(xr_bias_lin) # Todo ok!
    #
    #
    # # ===============================================================================
    # # Cold pool estimates
    # # ===============================================================================
    #
    #
    # path = Path(Framework.path)
    # stamod_removed_mean = train_stamod(path, remove_mean=True)
    # stamod = train_stamod(path, remove_mean=False)
    #
    #
    #
    # df_obs_pca_T = stamod['T']['stamod'].pca_reconstruct(range(1, 3))
    # df_obs_pca_Q = stamod['Q']['stamod'].pca_reconstruct(range(1, 3))
    # df_obs_pca_U = stamod['U']['stamod'].pca_reconstruct(range(1, 3))
    # df_obs_pca_V = stamod['V']['stamod'].pca_reconstruct(range(1, 3))
    #
    # df_obs_pca_T = df_obs_pca_T[1] + df_obs_pca_T[2] #+ df_obs_pca_T[3] + df_obs_pca_T[4]
    # df_obs_pca_Q = df_obs_pca_Q[1] + df_obs_pca_Q[2]
    # df_obs_pca_U = df_obs_pca_U[1] + df_obs_pca_U[2]
    # df_obs_pca_V = df_obs_pca_V[1] + df_obs_pca_V[2]
    #
    # #
    # #
    # # df_obs_pca = stamod['T']['stamod'].pca_reconstruct(range(1, 3))
    # # df_obs_pca = df_obs_pca[1] + df_obs_pca[2]
    # #
    # #
    # # df_obs = stamod['T']['stamod'].df.loc[stamod_removed_mean['T']['res']['lin']['index'], :]
    # # df_obs = df_obs_pca
    #
    #
    # df_lin_removed_mean = pd.DataFrame(stamod_removed_mean['T']['res']['lin']['predicted'].sum(axis=2),
    #                                index=stamod_removed_mean['T']['res']['lin']['index'], columns = df_obs_pca_T.columns)
    #
    # df_linq_removed_mean = pd.DataFrame(stamod_removed_mean['T']['res']['lin-q']['predicted'].sum(axis=2),
    #                                index=stamod_removed_mean['T']['res']['lin-q']['index'], columns = df_obs_pca_T.columns)
    #
    # df_dnn_removed_mean = pd.DataFrame(stamod_removed_mean['T']['res']['dnn']['predicted'].sum(axis=2),
    #                                index=stamod_removed_mean['T']['res']['dnn']['index'], columns = df_obs_pca_T.columns)
    #
    #
    # df_lin = pd.DataFrame(stamod['T']['res']['lin']['predicted'].sum(axis=2),
    #                                index=stamod['T']['res']['lin']['index'], columns = df_obs_pca_T.columns)
    #
    # df_linq = pd.DataFrame(stamod['T']['res']['lin-q']['predicted'].sum(axis=2),
    #                                index=stamod['T']['res']['lin-q']['index'], columns = df_obs_pca_T.columns)
    #
    # df_dnn = pd.DataFrame(stamod['T']['res']['dnn']['predicted'].sum(axis=2),
    #                                index=stamod['T']['res']['dnn']['index'], columns = df_obs_pca_T.columns)
    #
    # sc = stamod['T']['stamod'].scores[2]
    # idx_coldpool = sc.loc[sc<-15].index
    #
    #
    # df_gcm_t_m = pd.concat([df_gcm_t for d in range(len(df_obs_pca_T.columns))],axis=1)
    # df_gcm_t_m.columns = df_obs_pca_T.columns
    # df_lin_removed_mean = df_lin_removed_mean + df_gcm_t_m
    # df_dnn_removed_mean = df_dnn_removed_mean + df_gcm_t_m
    # df_linq_removed_mean = df_linq_removed_mean + df_gcm_t_m
    #
    #
    # stanames = ['rib_C10']
    #
    #
    # models = [df_lin, df_dnn, df_lin_removed_mean, df_dnn_removed_mean, df_gcm_t_m, df_linq,df_linq_removed_mean]
    #
    # dfs = []
    # for model in models:
    #     bias = model.subtract(df_obs_pca_T,axis=0)
    #     dfs.append(bias.loc[:,stanames].mean(axis=1))
    #
    #
    #
    # dfs = pd.concat(dfs,axis=1)
    #
    # dfs.columns = ['lin','dnn','lin-vies','dnn-vies','GCM raw','linq','linq_vies']
    #
    # print('NB of events of cold pool with PC2 >15 :'+ str(len(idx_coldpool)))
    # print('GCM raw : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'GCM raw']))))
    # print('linear model : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'lin']))))
    # print('linq : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'linq']))))
    # print('DNN model : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'dnn']))))
    #
    # print('linear model on GCM error : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'lin-vies']))))
    # print('linq on GCM error : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'linq_vies']))))
    # print('DNN model on GCM error : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'dnn-vies']))))
    #
    #
    # # ===============================================================================
    # #
    # # ===============================================================================
    #
    #
    #
    # df_lin_removed_mean = pd.DataFrame(stamod_removed_mean['Q']['res']['lin']['predicted'].sum(axis=2),
    #                                index=stamod_removed_mean['Q']['res']['lin']['index'], columns = df_obs_pca_Q.columns)
    #
    # df_linq_removed_mean = pd.DataFrame(stamod_removed_mean['Q']['res']['lin-q']['predicted'].sum(axis=2),
    #                                index=stamod_removed_mean['Q']['res']['lin-q']['index'], columns = df_obs_pca_Q.columns)
    #
    # df_dnn_removed_mean = pd.DataFrame(stamod_removed_mean['Q']['res']['dnn']['predicted'].sum(axis=2),
    #                                index=stamod_removed_mean['Q']['res']['dnn']['index'], columns = df_obs_pca_Q.columns)
    #
    #
    # df_lin = pd.DataFrame(stamod['Q']['res']['lin']['predicted'].sum(axis=2),
    #                                index=stamod['Q']['res']['lin']['index'], columns = df_obs_pca_Q.columns)
    #
    # df_linq = pd.DataFrame(stamod['Q']['res']['lin-q']['predicted'].sum(axis=2),
    #                                index=stamod['T']['res']['lin-q']['index'], columns = df_obs_pca_Q.columns)
    #
    # df_dnn = pd.DataFrame(stamod['Q']['res']['dnn']['predicted'].sum(axis=2),
    #                                index=stamod['Q']['res']['dnn']['index'], columns = df_obs_pca_Q.columns)
    #
    # sc = stamod['Q']['stamod'].scores[2]
    # idx_coldpool = sc.loc[sc>2.5].index
    #
    #
    # df_gcm_q_m = pd.concat([df_gcm_q for d in range(len(df_obs_pca_Q.columns))],axis=1)
    # df_gcm_q_m.columns = df_obs_pca_q.columns
    # df_lin_removed_mean = df_lin_removed_mean + df_gcm_q_m
    # df_dnn_removed_mean = df_dnn_removed_mean + df_gcm_q_m
    # df_linq_removed_mean = df_linq_removed_mean + df_gcm_q_m
    #
    #
    # stanames = ['rib_C10']
    #
    #
    # models = [df_lin, df_dnn, df_lin_removed_mean, df_dnn_removed_mean, df_gcm_t_m, df_linq,df_linq_removed_mean]
    #
    # dfs = []
    # for model in models:
    #     bias = model.subtract(df_obs_pca_Q,axis=0)
    #     dfs.append(bias.loc[:,stanames].mean(axis=1))
    #
    #
    #
    # dfs = pd.concat(dfs,axis=1)
    #
    # dfs.columns = ['lin','dnn','lin-vies','dnn-vies','GCM raw','linq','linq_vies']
    #
    # print('NB of events of wet valley with PC2 >2.5 :'+ str(len(idx_coldpool)))
    # print('GCM raw : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'GCM raw']))))
    # print('linear model : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'lin']))))
    # print('linq : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'linq']))))
    # print('DNN model : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'dnn']))))
    #
    # print('linear model on GCM error : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'lin-vies']))))
    # print('linq on GCM error : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'linq_vies']))))
    # print('DNN model on GCM error : ' + str(np.mean(np.abs(dfs.loc[idx_coldpool,'dnn-vies']))))
    #
    #
    #
    # plt.hist(dfs.loc[:,'lin'],alpha=0.4, label='lin', bins=20)
    # plt.hist(dfs.loc[:,'lin-vies'],alpha=0.4, label='lin-vies', bins=20)
    # plt.hist(dfs.loc[:,'dnn'],alpha=0.4, label='dnn', bins=20)
    # plt.hist(dfs.loc[:,'dnn-vies'],alpha=0.4, label='dnn-vies', bins=20)
    # plt.legend()
    # plt.show()
    #
    # # #
    # #
