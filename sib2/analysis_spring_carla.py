import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob


import geopandas
from rasterio import features
from affine import Affine
import numpy as np
from eofs.xarray import Eof


import os
# import xray
import matplotlib.pyplot as plt
# %matplotlib inline
import os
import rpy2.robjects as robjects

from rpy2.robjects import pandas2ri

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

from maihr_config import Rib_conf, Maihr_Conf
from mailib.toolbox.tools import common_index

import dask

from multiprocessing.pool import ThreadPool
dask.config.set(pool=ThreadPool(5))



def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def rasterize(shapes, coords, latitude='latitude', longitude='longitude',
              fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.

    usage:
    -----
    1. read shapefile to geopandas.GeoDataFrame
          `states = gpd.read_file(shp_dir)`
    2. encode the different shapefiles that capture those lat-lons as different
        numbers i.e. 0.0, 1.0 ... and otherwise np.nan
          `shapes = (zip(states.geometry, range(len(states))))`
    3. Assign this to a new coord in your original xarray.DataArray
          `ds['states'] = rasterize(shapes, ds.coords, longitude='X', latitude='Y')`

    arguments:
    ---------
    : **kwargs (dict): passed to `rasterio.rasterize` function

    attrs:
    -----
    :transform (affine.Affine): how to translate from latlon to ...?
    :raster (numpy.ndarray): use rasterio.features.rasterize fill the values
      outside the .shp file with np.nan
    :spatial_coords (dict): dictionary of {"X":xr.DataArray, "Y":xr.DataArray()}
      with "X", "Y" as keys, and xr.DataArray as values

    returns:
    -------
    :(xr.DataArray): DataArray with `values` of nan for points outside shapefile
      and coords `Y` = latitude, 'X' = longitude.


    """
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))

def get_shp_masks(ds, folder_shp):
    shape_files = glob.glob(folder_shp + '*.shp')

    for shapefile in shape_files:
        shp = geopandas.read_file(shapefile)
        # Reproject coordinate system of shape file
        shp = shp.to_crs({'init': 'epsg:4326'})

        shapes = (zip(shp.geometry, range(len(shp))))
        ds[os.path.basename(shapefile).split('.')[0]] = rasterize(shapes, ds.coords, longitude='lon',
                                                                  latitude='lat')
    return ds

def get_ds_w(path_w):

    robjects.r['load'](path_w)

    rdf = robjects.r['teste']

    dfs = []
    for i in range(1, len(rdf)):
        pd_df = pandas2ri.ri2py_dataframe(rdf[i])
        idx = pd.DatetimeIndex(pd_df.iloc[:, 0])
        idx = idx.tz_localize(None)
        pd_df.index = idx
        pd_df.index.name = 'time'
        pd_df = pd_df.iloc[:, 1:]
        pd_df = pd_df.loc[~pd_df.index.duplicated(keep='first')]

        dfs.append(pd_df)

    ds_w = xr.concat([df.to_xarray() for df in dfs], dim='sensor')
    ds_w.time.values = pd.DatetimeIndex(ds_w.time.values)
    sensor_names = list(rdf.names[1:])
    ds_w = ds_w.assign_coords(sensor=sensor_names)
    ds_w = ds_w.to_array(dim='depth')

    return ds_w

def get_ds_spring_sib2():
    """
    Return a dataset with the sib2 variables average for each Posses spring
    :return:
    """
    ds_spring_sib2 = []
    for var in list(ds_sib2.data_vars):  # TODO WARNING
        print('=' * 20)
        print(var)
        print('=' * 20)

        ds_sib2_var = ds_sib2[var].load()

        ds_sib2_var = get_shp_masks(ds_sib2_var, folder_shp)
        spring_names = list(ds_sib2_var.coords)[len(ds_sib2_var.dims) + 1:]

        ds_springs = []
        for spring_name in spring_names:
            ds_spring = ds_sib2_var.where(ds_sib2_var[spring_name] == 0)
            ds_spring = ds_spring.mean(dim=['lat', 'lon'])
            ds_springs.append(ds_spring)
        ds_springs = xr.concat(ds_springs, dim='spring')
        ds_springs = ds_springs.assign_coords(spring=spring_names)
        ds_springs = ds_springs.to_dataset(name=var)
        ds_spring_sib2.append(ds_springs)


    ds_spring_sib2 = xr.merge(ds_spring_sib2)
    ds_spring_sib2.to_netcdf(str(conf.sib / "pontual_ribeirao/out/ds_sib2_springs.nc"))

    return ds_spring_sib2

if __name__ == '__main__':

    conf = Rib_conf()
    conf_mair = Maihr_Conf()
    # #####################
    # # Read Sib2 data
    # #####################
    # # dap = xr.open_dataset("/home/thomas/computer/springs/sib2_ribdata1_pastagem.nc")
    # # dap = xr.open_dataset("/home/thomas/computer/springs/out_nc/sib2_ribdata1_2049.nc")
    # # ds_sib2 = xr.open_dataset(str(conf.sib / "out/clim_2014.nc"))
    #
    # fig_output_path = str(conf.sib / "pontual_ribeirao/fig/")
    # # ds = xr.open_dataset("/home/thomas/pro/research/framework/stamod_rib/out/clim_2014.nc")
    # infolder = str(conf.sib / 'pontual_ribeirao/out_nc')+'/*.nc'
    # print(infolder)
    # ds_sib2 = xr.open_mfdataset(infolder)
    #
    # ds_sib2 = ds_sib2.chunk(1500)
    #
    # _, index = np.unique(ds_sib2['time'], return_index=True)
    # ds_sib2 = ds_sib2.isel(time=index)
    # ds_sib2 = ds_sib2.reindex({'time': pd.DatetimeIndex(ds_sib2.time.values)})
    # ds_sib2 = ds_sib2.transpose('time', 'lat', 'lon')
    #
    # ###################
    # # Get spring shapes mask
    # ##################
    # folder_shp = str(conf_mair.framework_path / "stamod_rib/data/shapes_nascentes_calra/area_individuais_DRONE/")+'/'
    # print(folder_shp)
    #
    # ds_spring_sib2 = get_ds_spring_sib2()
    # print(ds_spring_sib2)
    #

    # Select by shape


    nc_file_path = "/home/thomas/database_outdropbox/sib2/ds_sib2_springs.nc"
    ds_spring_sib2 = xr.open_dataset(nc_file_path)
    ds_spring_sib2.time.values= pd.DatetimeIndex(ds_spring_sib2.time.values) - pd.DateOffset(years=50)


    #
    # ###################
    # # Get Soil humidity Data
    # ##################
    # path_w = "/home/thomas/pro/resea  rch/framework/stamod_rib/data/UmidadeSolo_Posses_abril2019_hora.RData"
    # ds_w = get_ds_w(path_w)
    #
    ###################
    # Get Spring Data
    ##################
    df_spring = pd.read_csv("/home/thomas/pro/research/framework/stamod_rib/data/spring/vazao_geral_mmD.csv",
                            index_col=0, parse_dates=True, sep=';')

    df_posses = pd.read_csv('/home/thomas/pro/research/framework/stamod_rib/data/spring/FozPosses_todo_periodo_baseflow_bsf_1.txt',
                            index_col=0, parse_dates=True, delim_whitespace=True)


    df_spring = df_spring.drop(columns = ['R21','NA09'])


    df_spring = df_spring.loc['2016-10':]
    df_spring = df_spring.loc[:,df_spring.count()>25]
    df_spring = df_spring.dropna(axis=0,how='any')

    df_posses = df_posses.loc['2016-10':]

    ds_spring = df_spring.to_xarray()
    ds_posses = df_posses.to_xarray()

    ds_spring = ds_spring.rename({'date':'time'})
    ds_posses = ds_posses.rename({'date':'time'})

    _, index = np.unique(ds_spring['time'], return_index=True)
    ds_spring = ds_spring.isel(time=index)
    ds_spring = ds_spring.reindex({'time': pd.DatetimeIndex(ds_spring.time.values)})

    ds_spring = ds_spring.to_array(dim='spring')
    ds_spring = ds_spring.transpose('time', 'spring')

    ds_spring.time.values = pd.DatetimeIndex(ds_spring.time.values)
    ds_spring.name = 'obs'


    # fig_output_path ='/home/thomas/pro/research/framework/stamod_rib/fig/'
    # plot_pca_analysis(ds_spring, fig_output_path, title='spring')


    #####################################
    # Analyse Observed and estimated spring runoff
    #####################################

    var = 'Evpt'
    ds_spring_sib2 = ds_spring_sib2[var]
    ds_spring_sib2 = ds_spring_sib2.resample(time='M').min(dim='time')
    ds_spring_sib2 = ds_spring_sib2.transpose('time','spring')

    ds_spring_sib2 = ds_spring_sib2.dropna(dim="spring", how="all")
    ds_spring_sib2 = ds_spring_sib2.dropna(dim="time", how="all")
    ds_spring_sib2 = ds_spring_sib2.dropna(dim="spring", how="any")

    idx = common_index(ds_spring_sib2.spring.values, ds_spring.spring.values)

    ds_spring_sib2 = ds_spring_sib2.sel(spring=idx)
    ds_spring = ds_spring.sel(spring=idx)


    plt.scatter(ds_spring_sib2.min('time').to_dataframe(), ds_spring.min('time').to_dataframe())



    #######################
    # PCA Analysis
    #######################

    nbpcs = 3

    solver_sib2 = Eof(ds_spring_sib2.dropna(dim="time", how="all"))
    pcas_sib2 = solver_sib2.pcs(npcs=nbpcs, pcscaling=1)
    eofs_sib2 = solver_sib2.eofs(neofs=nbpcs, eofscaling=1)

    solver = Eof(ds_spring.dropna(dim="time", how="all"))
    pcas = solver.pcs(npcs=nbpcs, pcscaling=1)
    eofs = solver.eofs(neofs=nbpcs, eofscaling=1)

    fig, axes = plt.subplots(3, 4, figsize=(20, 20))
    pcas.to_dataframe().unstack().plot(ax=axes[0,0])
    pcas_sib2.to_dataframe().unstack().plot(ax=axes[0,1])

    df_eofs = eofs.to_dataframe().unstack().T
    df_eofs_sib2 = eofs_sib2.to_dataframe().unstack().T
    df_eofs.index = df_eofs.index.levels[1]
    df_eofs_sib2.index = df_eofs_sib2.index.levels[1]


    df_eofs.plot(ax=axes[1,0])
    df_eofs_sib2.plot(ax=axes[1,1])

    plt.scatter(df_eofs.iloc[:,2],df_eofs_sib2.iloc[:,1])

    plt.show()
    print('done')


    # Read predictores Carla
    df_pred = pd.read_csv('/home/thomas/pro/research/framework/stamod_rib/data/predictores_spring_carla.csv',
                          index_col=0, parse_dates=True)
    df_pred = df_pred.drop(columns='solo predominante')


    dd = pd.concat([df_eofs, df_eofs_sib2, df_pred], join='inner', axis=1)
