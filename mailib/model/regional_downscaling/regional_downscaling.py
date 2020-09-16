"""
DESCRIPTION

    Perform the downscaling of the near-surface climate in the regional domain.
    Correcting GFS temperature free air-lapse rate at the DEM surface with spatial and temporal predictors.
"""

from stadata.lib.LCBnet_lib import *
import matplotlib.pyplot as plt
from toolbox import common_index
from sklearn import preprocessing, model_selection,metrics
from sklearn.ensemble import RandomForestRegressor
import pickle
from scipy.optimize import curve_fit


def model_CAD(df_t, df_gfs_t, plot=False):
    new_df_t = df_t.loc["2015-07-30 00:00:00":"2015-08-15 00:00:00",:]
    new_df_t.append(df_t.loc["2015-08-02 00:00:00":"2015-08-19 00:00:00",:])
    df_t = new_df_t

    df_bias_t = df_t.sub(df_gfs_t.loc[df_t.index,df_t.columns], axis='columns')
    df_bias_t.dropna(inplace=True, axis=0, how='all')
    df_bias_t.dropna(inplace=True, axis=1, how='all')

    df_bias_t = df_bias_t.mean()

    AttSta_predictors = get_spatial_predictors()

    df_X = AttSta_predictors.attributes.loc[df_bias_t.index,['8','5','3']] # good set!

    df_X.dropna(inplace=True, axis=0, how='any')

    df_bias_t = df_bias_t[df_X.index]
    X = df_X
    y = df_bias_t

    #===========================================================================
    # Prepocessing Standardisation
    #====  =======================================================================
    # X= preprocessing.scale(X)
    X_norm = X
    #===========================================================================
    # Create Training and Test sets
    #====  =======================================================================
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_norm, y, test_size=0.2)#, random_state=0)

    #===========================================================================
    # Random Forest
    #====  =======================================================================

    Rf = RandomForestRegressor(
                            n_estimators=500,
                            criterion='mse',
                            max_depth=2,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features=2,
                            max_leaf_nodes=None,
                            bootstrap=True,
                            oob_score=False,
                            n_jobs=1,
    )

    print('Fiting ..')
    Rf.fit(X_train, y_train)
    print('Finished')

    #===========================================================================
    # Feature importance
    print('Feature importance')
    #===========================================================================
    df_feature_importance = pd.DataFrame(Rf.feature_importances_, index = X.columns)
    print(df_feature_importance.sort_values(0, ascending=False))

    #===========================================================================
    # Model Validation
    print('Model Validation')
    #===========================================================================

    mae = metrics.mean_absolute_error(y_train.values, Rf.predict(X_train))
    print('Train MAE GFS: '+str(mae))
    mae = metrics.mean_absolute_error(y_test.values, Rf.predict(X_test))
    print('Test MAE GFS: '+str(mae))
    mse = metrics.mean_squared_error(y_train.values, Rf.predict(X_train))
    print('Train MSE GFS: '+str(mse))
    mse = metrics.mean_squared_error(y_test.values, Rf.predict(X_test))
    print('Test MSE GFS: '+str(mse))

    #===========================================================================
    # Plot
    print('Model Validation')
    #===========================================================================
    if plot:
        f = plt.figure()
        plt.plot(y_test.values)
        plt.plot(Rf.predict(X_test))
        plt.show()

        df_full = pd.concat([y_train,pd.DataFrame(Rf.predict(X_train), index=X_train.index) ], axis=1,join='inner')
        df_full.columns=['observed', 'predicted']
        df_full.plot()
        plt.show()

        ax = df_full.plot(kind='scatter', x='observed',y='predicted')

        for sta in df_full.index:
            ax.annotate(sta,xy=(df_full.loc[sta,'observed'], df_full.loc[sta,'predicted']))

        plt.show()

    return Rf

def get_data_temporal_modeling(df_t, df_gfs_t, df_cad):

    df_bias_t = df_t.sub(df_gfs_t.loc[df_t.index,df_t.columns], axis='columns')
    df_bias_t.dropna(inplace=True, axis=0, how='all')
    df_bias_t.dropna(inplace=True, axis=1, how='all')


    #===========================================================================
    # Get Data
    #====  =======================================================================
    stanames = common_index(df_cad.index,df_bias_t.columns )

    # get same spatial index
    df_cad = df_cad.loc[stanames]
    df_bias_t = df_bias_t.loc[:,stanames]

    df_ys = []
    df_Xs = []
    df_indexs = []

    for staname in stanames:
        # get daata
        df_y = df_bias_t.loc[:, staname] # get the response
        df_temporal_predictor = pd.read_csv(path_gfs_predictors + 'gfs025_'+staname+'.csv', index_col=0, parse_dates=True) # get the predictors at the station position

        # select temporal predictors
        df_temporal_predictor = df_temporal_predictor.loc[:,['Rh500mb']]

        # Get same temporal index
        temporal_index = pd.to_datetime(common_index(df_temporal_predictor.index, df_y.index)) # get the same index
        df_y = df_y.loc[temporal_index]
        df_temporal_predictor = df_temporal_predictor.loc[temporal_index,:]

        # reformat CAD
        df_spatial_predictor_cad= pd.DataFrame(np.repeat([df_cad.loc[staname].values], len(temporal_index),axis=0), columns =['CAD'], index= temporal_index )

        # Concat the predictors
        df_X = pd.concat([df_spatial_predictor_cad,df_temporal_predictor], axis=1)
        df_index=pd.DataFrame([temporal_index, np.repeat(staname, len(temporal_index))], index=['time', 'staname']).T

        df_ys.append(df_y)
        df_Xs.append(df_X)
        df_indexs.append(df_index)

    df_index = pd.concat(df_indexs,axis=0, join='inner',ignore_index=True)
    df_X = pd.concat(df_Xs,axis=0, join='inner',ignore_index=True)
    df_y = pd.concat(df_ys,axis=0, join='inner',ignore_index=True)

    return df_X, df_y, df_index

def get_spatial_predictors():
    """
    Get all the attributes at each stations

    :return:
    """
    # Set attributes
    # AttSta = att_sta("/home/thomas/phd/obs/staClim/metadata/database_metadata.csv")
    # AttSta.attributes['Alt'] = AttSta.attributes['Alt'].astype(float)
    # AttSta.attributes = AttSta.attributes.loc[:,['Lat','Lon','Alt']]
    # AttSta.attributes.dropna(how='all',axis=1, inplace=True)
    # AttSta.attributes.dropna(how='any',axis=0, inplace=True)
    # print AttSta.attributes

    # Measured and calculated attributes for temperature spatial variability
    AttSta = att_sta('/home/thomas/phd/framework/predictors/out/att_sta/df_topoindex_regional.csv')
    AttSta.attributes['Alt'] = AttSta.attributes['Alt'].astype(float) # if not give problem in the stepwise libnear regression
    AttSta.attributes.loc[AttSta.attributes['Alt'].isnull(),'Alt'] = AttSta.attributes.loc[AttSta.attributes['Alt'].isnull(),'alt_dem'] # replace cetesb no Alt value by alt_dem

    # AttSta_index.attributes.dropna(how='all',axis=1, inplace=True)
    AttSta.attributes.dropna(how='any',axis=1, inplace=True)
    AttSta.addatt(df= AttSta.attributes)

    # Place alt_dem where their is no Alt
    # remove duplicated attributes
    AttSta.attributes = AttSta.attributes.loc[:,~AttSta.attributes.columns.duplicated()]

    # # # # DEM extracted attributes
    # AttSta_topex = att_sta('/home/thomas/phd/framework/predictors/out/att_sta/sta_att_distancetosea.csv')
    # AttSta_topex.attributes = AttSta_topex.attributes.convert_objects(convert_numeric=True)
    # AttSta.addatt(df= AttSta_topex.attributes)
    # #
    # # # remove duplicated attributes
    # AttSta.attributes = AttSta.attributes.loc[:,~AttSta.attributes.columns.duplicated()]


    # AttSta.attributes.drop('alt_dem',axis=1, inplace=True)
    # AttSta.attributes.drop('Alt',axis=1, inplace=True)
    # AttSta.attributes.drop('Lat',axis=1, inplace=True)
    # AttSta.attributes.drop('Lon',axis=1, inplace=True)

    # AttSta.attributes = AttSta.attributes[AttSta.attributes.loc[:,'Alt']>500]

    AttSta.attributes = AttSta.attributes._get_numeric_data() # get only numeric data

    return AttSta


temporal_predictor_selected  = ['T900mb','T850mb','T500mb','U900mb','U850mb','U500mb','V900mb','V850mb','V500mb','Q2m','Q80m']
temporal_predictor_selected  = ['T2m','Q2m']

spatial_predictor_selected = ['rsp64 ','diss64']


if __name__=='__main__':
    #===========================================================================
    # Data
    #====  =======================================================================
    path_gfs_predictors = "/home/thomas/phd/framework/predictors/out/gcm_data/regional/by_sta/"
    path_gfs_interp_T = "/home/thomas/phd/framework/predictors/out/gcm_data/regional/interp_dem/"
    path_obs_T = "/home/thomas/phd/framework/fill/out/"

    # Interpolated GFS free air temperature
    df_gfs_t = pd.read_csv(path_gfs_interp_T + 'df_Ta.csv', index_col=0, parse_dates=True)
    df_gfs_t = df_gfs_t-273.15
    # df_gfs_t = df_gfs_t.between_time('03:00','03:00')

    # Observed data of temperature
    df_t = pd.read_csv(path_obs_T + "Ta.csv", index_col=0, parse_dates=True).resample('H').mean()
    df_t = df_t.between_time('03:00','03:00') # nighttime

    # GOOD SET CAD modeling!!!! MSE
    df_t.drop(['SBSJ','sb69','A706','SBPC','su81'], axis=1, inplace=True)
    df_t.drop(['A737','A715','A740'], axis=1, inplace=True)
    df_t.drop(['co40','A509'], axis=1, inplace=True)
    df_t.drop(['jn38'], axis=1, inplace=True)
    df_t.drop(['SBSC'], axis=1, inplace=True)
    df_t.drop(['pr141','36','51','tr82'], axis=1, inplace=True)

    #===========================================================================
    # Modeling CAD
    #====  =======================================================================
    # model_cad = model_CAD(df_t, df_gfs_t, plot=True)
    # pickle.dump( model_cad, open("/home/thomas/phd/framework/model/out/regional_model/"+"model_cad.p", "wb" ) )

    model_cad = pickle.load( open("/home/thomas/phd/framework/model/out/regional_model/"+"final_model_cad.p", "rb" ) )

    AttSta = get_spatial_predictors()
    df_X = AttSta.attributes.loc[df_t.columns,['8','5','3']]
    df_X.dropna(inplace=True, axis=0, how='any')
    df_cad = model_cad.predict(df_X)
    df_cad = pd.DataFrame(df_cad,index = df_X.index )
    # # df_y.plot()
    # # plt.show()

    df_bias_t = df_t.sub(df_gfs_t.loc[df_t.index,df_t.columns], axis='columns')
    df_bias_t.dropna(inplace=True, axis=0, how='all')
    df_bias_t.dropna(inplace=True, axis=1, how='all')

    #===========================================================================
    # Get temporal predictors
    #====  =======================================================================
    stanames = common_index(df_cad.index,df_bias_t.columns )


    def get_data(var, stanames):
        dfs_temporal = []
        for staname in stanames:
            df_temporal_predictor = pd.read_csv(path_gfs_predictors + 'gfs025_'+staname+'.csv', index_col=0, parse_dates=True)
            dfs_temporal.append(df_temporal_predictor.loc[:,var])
        dfs_temporal =  pd.concat(dfs_temporal, axis=1)
        dfs_temporal.columns = stanames
        return dfs_temporal


    dfs_gfs_rh500mb = get_data('Rh500mb', stanames)
    dfs_gfs_rh500mb = get_data('Rh900mb', stanames)
    dfs_gfs_q2m = get_data('Q2m', stanames)


    temporal_index = pd.to_datetime(common_index(dfs_gfs_rh500mb.index, df_bias_t.index))
    dfs_gfs_rh500mb = dfs_gfs_rh500mb.loc[temporal_index,stanames]
    df_bias_t = df_bias_t.loc[temporal_index,stanames]
    df_gfs_t = df_gfs_t.loc[temporal_index,stanames]
    df_t = df_t.loc[temporal_index,stanames]


    df_spatial_predictor_cad= pd.DataFrame(np.repeat([df_cad.values.flatten()], len(df_bias_t.index),axis=0), index= temporal_index, columns =df_cad.index )


    xdata = pd.concat([df_spatial_predictor_cad.stack(), dfs_gfs_rh500mb.stack()], axis=1)
    xdata= pd.DataFrame(preprocessing.scale(xdata), columns = xdata.columns, index=xdata.index)

    ydata = df_bias_t.stack()

    def multi_lin(X, a, b,c,d):
        x0,x1 = X
        return a*x0 + b*x1+ c*x0*x1 +d

    x = xdata.values
    y = ydata.values.reshape(-1,1)

    popt, pcov = curve_fit(multi_lin,(xdata.iloc[:,0].values,xdata.iloc[:,1].values ) ,y.flatten())

    df_y_data = ydata.unstack()

    Y_predicted = multi_lin((xdata.iloc[:,0].values,xdata.iloc[:,1].values ),*popt )

    df_Y_predicted = pd.DataFrame(Y_predicted, index = ydata.index)
    df_Y_predicted = df_Y_predicted.unstack(level=-1)
    df_Y_predicted.columns = df_y_data.columns
    df_Y_predicted.index = df_y_data.index



    df_check = df_gfs_t.add(df_y_data, axis='columns')

    dfmyfuckingreconstruvtiondcscsdcs = df_gfs_t.add(df_Y_predicted, axis='columns')

    # ax = df_check.loc[:,['C10','C09']].plot()
    ax = df_t.loc[:,['C10','C09']].plot()
    df_gfs_t.loc[:,['C10','C09']].plot(ax=ax)
    dfmyfuckingreconstruvtiondcscsdcs.loc[:,['C10','C09']].plot(ax=ax)


    plt.plot(y,'b')
    plt.plot(Y_predicted,'g')
    plt.show()


    mae = metrics.mean_absolute_error(df_t.values.flatten(), df_gfs_t.values.flatten())
    print('MAE by GFS Interpolation: '+str(mae))
    mae = metrics.mean_absolute_error(df_t.values.flatten(), dfmyfuckingreconstruvtiondcscsdcs.values.flatten())
    print('MAE by my downscaling model GFS: '+str(mae))


    mae = metrics.mean_squared_error(df_t.values.flatten(), df_gfs_t.values.flatten())
    print('MSE by GFS Interpolation: '+str(mae))
    mae = metrics.mean_squared_error(df_t.values.flatten(), dfmyfuckingreconstruvtiondcscsdcs.values.flatten())
    print('MSE by my downscaling model GFS: '+str(mae))



    #===========================================================================
    # RECONSTRUCTION OF THE DOWNSCALED FIELD
    #====  =======================================================================
    # -> gridded ['8','5','3'] predictors
    # -> gridded [Rh900mb]
    # -> grided temperature interp [for 1day]


    arr_rh900mb = np.load("/home/thomas/phd/framework/predictors/out/df_field/regional/field_Rh900mb_S23W047_2015081106.npy")
    arr_T_interp = np.load("/home/thomas/phd/framework/predictors/out/gcm_data/regional/interp_dem/fields/field_T_S23W047_2015081106.npy")



    print('Done')
