"""
DESCRIPTION
    Implement Autoencoder for climate prediction in Ribeirao Das Posses


"""

# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import GRU
from keras.models import Sequential
from nn_local import get_datasets_ribeirao
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer

from toolbox import common_index

# Get Data



if __name__ == '__main__':

    # User path
    path_main = "/home/thomas/Documents/posdoc/framework" # To be adapted on your local machine
    path_predictors = '/predictors/out/gcm_data/local/basic_var/df_features_scores.csv'
    path_response = '/fill/out/Ta.csv'
    path_model_save = '/model/out/nn/autoencoder_'


    # ===========================================================================
    # Get response
    # ===========================================================================
    stadata = pd.read_csv(path_main + path_response, index_col=0, parse_dates=True)
    # stadata = stadata.filter(regex='C', axis=1)[:-2]
    stadata.dropna(inplace=True, axis=0, how='all')
    stadata.dropna(inplace=True, axis=1, how='all')
    pca = PCA(4)
    scores = pca.fit_transform(stadata)

    df_y = pd.DataFrame(scores[:, 0], index= stadata.index)

    # ===========================================================================
    # Get GFS data
    # ===========================================================================
    path_gfs_interp = "/home/thomas/Documents/posdoc/framework/" \
                      "predictors/out/gcm_data/regional/interp_dem/"
    df_gfs = pd.read_csv(path_gfs_interp+"df_Ta.csv",parse_dates=True, index_col=0)
    df_gfs = df_gfs - 275.15 # convert in degree
    df_gfs = df_gfs[df_gfs.index.hour==15]
    idxs_col = common_index(stadata.columns, df_gfs.columns)

    # Bias
    idxs_row = common_index(stadata.index, df_gfs.index)
    bias_T = stadata.loc[idxs_row, idxs_col] - df_gfs.loc[idxs_row, idxs_col]

    # ===========================================================================
    # Get df_topoindex regional
    # ===========================================================================
    path_topoindex = "/home/thomas/Documents/posdoc/framework/predictors/out/att_sta/"
    df_topoindex = pd.read_csv(path_topoindex+ "df_topoindex_regional.csv",index_col=0)
    df_topoindex = df_topoindex.loc[idxs_col]

    df_topoindex.dropna(inplace=True, axis=1, how='any')
    df_topoindex.dropna(inplace=True, axis=1, how='all')

    # ===========================================================================
    # Get response
    # ===========================================================================

    stadata = pd.read_csv(path_main + path_response, index_col=0, parse_dates=True)
    # stadata = stadata.filter(regex='C', axis=1).iloc[:,:-2]

    stadata_daily = [stadata.shift(hour) for hour in range(24)]

    stadata_daily = pd.concat(stadata_daily, axis=1, join='inner')
    stadata_daily = stadata_daily[stadata_daily.index.hour == 0]

    stadata.dropna(inplace=True, axis=0, how='any')
    stadata.dropna(inplace=True, axis=1, how='all')

    stadata_daily.dropna(inplace=True, axis=0, how='any')
    stadata_daily.dropna(inplace=True, axis=1, how='all')

    scaler = Normalizer()
    stadata_norm = scaler.fit_transform(stadata)


    scaler_daily = Normalizer()
    stadata_norm = scaler_daily.fit_transform(stadata_daily)

    nb_pcs = 4
    pca = PCA(nb_pcs)
    scores = pca.fit_transform(stadata)
    df_scores = pd.DataFrame(scores, columns = range(nb_pcs),index=stadata.index)

    pca_daily = PCA(4)
    scores_daily = pca_daily.fit_transform(stadata_daily)
    df_scores_daily = pd.DataFrame(scores_daily, columns=range(nb_pcs), index=stadata_daily.index)

    pca_bias = PCA(10)
    df_scores_bias= pca_bias.fit_transform(bias_T)
    df_scores_bias = pd.DataFrame(df_scores_bias, columns=range(10), index=bias_T.index)

    pca_bias.explained_variance_ratio_
    loadings = pca_bias.components_

    # Correlation analysis with loadings
    df_topoindex = pd.concat([df_topoindex,pd.DataFrame(loadings.T,index=df_topoindex.index,
                                                        columns=['PC'+ str(c) for c in range(loadings.T.shape[1])])],axis=1)




    idxs_select = df_topoindex.corr().index[df_topoindex.corr().loc[:, 'PC2'] > 0.6]

    fig, axes = plt.subplots(nrows=2,ncols=2)

    df_topoindex.corr().loc[:, 'PC0'].plot.bar(ax=axes[0,0])
    df_topoindex.corr().loc[:, 'PC1'].plot.bar(ax=axes[0,1])
    df_topoindex.corr().loc[:, 'PC2'].plot.bar(ax=axes[1,0])
    df_topoindex.corr().loc[:, 'PC3'].plot.bar(ax=axes[1,1])

    plt.show()

    var1 = PC0
    var2 = PC0

    def scatter_plot(var1, var2,axs):
        # scatter_matrix(df_topoindex.loc[:,idxs_select])
        axs.scatter(df_topoindex.loc[:, var1], df_topoindex.loc[:, var2])
        for idx in df_topoindex.index:
            axs.annotate(idx,xy=(df_topoindex.loc[idx, var1], df_topoindex.loc[idx, var2]) )
        axs.set_title(var1+var2)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    scatter_plot('PC0','PC1',axes[0,0])
    scatter_plot('PC0', 'PC2', axes[0, 1])
    scatter_plot('PC0', 'PC3', axes[1, 1])
    scatter_plot('PC0', 'PC4', axes[1, 0])

    plt.show()

    raster_val = np.loadtxt(path_main + "/home/thomas/phd/geomap/data/regional_raster/map_lon44_49_lat20_25_lowres", delimiter=',')
    raster_lon = np.loadtxt("/home/thomas/phd/geomap/data/regional_raster/map_lon44_49_lat20_25_lowreslongitude.txt",
                            delimiter=',')
    raster_lat = np.loadtxt("/home/thomas/phd/geomap/data/regional_raster/map_lon44_49_lat20_25_lowreslatitude.txt",
                            delimiter=',')

    AttSta = att_sta('/home/thomas/phd/obs/staClim/metadata/database_metadata.csv')
    stalatlon = AttSta.attributes.loc[:, ['Lat', 'Lon', 'network']]

    plt, map = map_domain(raster_lat, raster_lon, raster_val)
    map = add_stations_positions(map, stalatlon)
    plt.legend(loc='best', framealpha=0.4)

    # plt.show()
    plt.savefig("/home/thomas/phd/climaobs/res/map/map_stations_domain_regional.eps", transparent=True, dpi=500)
    #

    # Get datasets
    df_x, df_y = get_datasets_ribeirao()

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_x_scaled = scaler.fit_transform(df_x.values)
    df_y_scaled = scaler.fit_transform(df_y.values)

    # Spliting sets
    x_train, x_test, y_train, y_test = train_test_split(df_x_scaled, df_y_scaled, test_size=0.10, random_state=1)

    # Select model type
    model = Sequential()

    model.add(GRU())



