"""

    SIB2: Land surface model

"""

import pandas as pd
import glob

from shutil import copyfile
import os
import xarray as xr
import numpy as np
import luigi
import dask
import matplotlib.pyplot as plt
from mailib.toolbox.geo import xr_cart2pol,cart2pol
from mailib.toolbox.meteo import Ev
from sibcat_conf import Rib_conf #, Maihr_Conf # To export to rodolfo
from downscaling_ribeirao.statmod_articleII import predict_stamod_rib, predict_csf_posses

from dateutil.relativedelta import relativedelta

# from multiprocessing.pool import ThreadPool
# dask.config.set(pool=ThreadPool(20))

def irr_rib_df_to_xr_rib(out_irr_nc):
    print('start get array')

    conf = Maihr_Conf()
    framework_path  = conf.framework_path

    # Irr
    irr_file_path = "database/predictors/out/irr/df_irr_rib_year.csv"
    lat_irr_file_path = 'database/predictors/out/irr/lat.csv'
    lon_irr_file_path = 'database/predictors/out/irr/lon.csv'

    print('start get irr')
    # Get Irr data # The fact that grass run on python 2.7 fucked everything
    df_irr = pd.read_csv(str(framework_path / irr_file_path), index_col=0)
    df_irr_lat = pd.read_csv(str(framework_path / lat_irr_file_path), index_col=0)
    df_irr_lon = pd.read_csv(str(framework_path / lon_irr_file_path), index_col=0)
    print('finish read csv')

    lat = df_irr_lat.values[:].reshape(1, -1)[0]  # todo Feio
    lon = df_irr_lon.values[:].reshape(1, -1)[0]
    irr_dates = pd.to_datetime(df_irr.columns)

    irr_dates = irr_dates + pd.offsets.DateOffset(years=100)  # Todo: This could be cleaner

    print('Reshape from dataframe to xarray')
    shape_ribeirao = (239, 245)  # Todo this must be generalized
    xrs = []
    for i in range(len(df_irr.columns)):
        xrs.append(xr.DataArray(df_irr.values[:, i].reshape(shape_ribeirao), coords={'lat': lat, 'lon': lon},dims=['lat', 'lon']))
    xr_irr = xr.concat(xrs, dim='time')
    xr_irr = xr_irr.assign_coords(time=irr_dates)

    # Elev
    elev_file_path = "database/predictors/out/irr/posses@PERMANENT"
    arr_elev = pd.read_csv(str(framework_path / elev_file_path), header=None)
    xr_irr.coords['elev'] = (('lat', 'lon'), arr_elev.values)

    # dates_hourly_irr = pd.date_range(xr_irr.time[0].values, xr_irr.time[-1].values, freq='H')
    # dates_hourly = pd.date_range(pd.to_datetime(str(year)), pd.to_datetime(str(year+1)), freq='H')

    print('interpolate')
    xr_irr.name = 'clim'
    # xr_irr = xr_irr.fillna(0) # DOES NOT WORK
    # xr_irr.values = xr_irr.values * 0.8  # Todo WARNING: does not assign with isel, it fail silently
    print('finish Irr')
    xr_irr.to_netcdf(out_irr_nc)
    xr_irr = xr_irr.to_dataset(name='clim')
    return xr_irr

def add_Rib_irr_and_interp(ds_clim, year):

    rib_conf = Rib_conf()
    irr_data_path = rib_conf.irr_data_path

    if os.path.exists(irr_data_path):
        ds_irr = xr.open_dataset(irr_data_path)
    else:
        ds_irr = irr_rib_df_to_xr_rib(irr_data_path) # TO make xr irr

    print('reindex hourly')
    dates_hourly = pd.date_range(str(pd.to_datetime(ds_irr.time.values).year[0]),str(pd.to_datetime(ds_irr.time.values).year[0]+1), freq='H', closed='left')
    ds_irr = ds_irr.reindex(time=dates_hourly) # It has a duplicated value I dont know why
    ds_irr.coords['time'] = pd.DatetimeIndex(ds_irr.time.values)  + pd.DateOffset(years=year-2000)
    ds_irr = ds_irr.interp(lat=ds_clim.lat, lon=ds_clim.lon, method='nearest')

    # Predict csf iir
    model_path = str(rib_conf.framework_path) + "/model/out/statmod/irr_rib_model/csf_model_v9.h5"
    df_pred_csf = predict_csf_posses(model_path)

    df_pred_csf = df_pred_csf.reindex(pd.date_range(df_pred_csf.index[0], df_pred_csf.index[-1], freq='H')) # Reindex hurly
    df_pred_csf = df_pred_csf.interpolate('slinear')
    df_pred_csf = df_pred_csf.loc[pd.to_datetime(ds_irr.time.values)].values[:, np.newaxis]

    ds_irr = ds_irr * df_pred_csf
    # Drop duplicates
    _, index = np.unique(ds_irr['time'], return_index=True)
    ds_irr = ds_irr.isel(time=index)


    dates_hourly = pd.date_range(str(pd.to_datetime(ds_irr.time.values).year[0]),str(pd.to_datetime(ds_irr.time.values).year[0]+1), freq='H', closed='left')

    # Drop duplicates
    _, index = np.unique(ds_clim['time'], return_index=True)
    ds_clim = ds_clim.isel(time=index)
    ds_clim = ds_clim.interp(time=dates_hourly, method='cubic')
    ds_clim = ds_clim.reindex(time=dates_hourly, method='nearest')

    ds_irr.coords['mask'] = (('lat', 'lon'), ds_clim['mask'].values)
    ds_irr = ds_irr.assign_coords(var='ki')

    T = ds_clim['clim'].sel(var='T') + 273.15 # TODO CANNOT assign in dask array
    T = T.to_dataset(name='clim')
    ds = ds_clim.drop("T", dim='var')

    ds_clim =  xr.concat([ds, T], dim='var')
    print('finished get clim')


    ds_clim.coords['elev'] =(('lat', 'lon'), ds_irr['elev'].values)
    ds =  xr.concat([ds_clim, ds_irr], dim='var')



    return ds

def make_data2nc_from_maihr(From_year, To_year):

    rib_conf = Rib_conf()
    ######################
    # Maihr TQUV
    ######################
    # xr_clim_paths = glob.glob(os.path.dirname(self.input().path)+'/clim_*.nc')
    xr_clim_paths = glob.glob(str(rib_conf.maihrdata_path))
    xr_clim = xr.open_mfdataset(xr_clim_paths)
    xr_clim = xr_clim.reindex(time=sorted(xr_clim.time.values))
    xr_clim = xr_clim.load()  # WARNING this could blow up the memmory
    # xr_clim = xr_clim.chunk(chunks={'lat': 10, 'lon': 10})
    # time.sleep(int(np.random.randint(120)))

    ######################
    # CPC Rain
    ######################
    # file_cpc = self.framework_path / 'stamod_rib/data/precip_1979a2017_CPC_AS.nc'
    file_cpc = str(rib_conf.framework_path /'stamod_rib/data/precip_1979a2017_CPC_AS.nc')
    xr_rain_cpc = xr.open_dataarray(str(file_cpc))
    xr_rain_cpc.coords['lon'].values = (xr_rain_cpc.coords['lon'].values + 180) % 360 - 180

    ###################
    # Add KI and Interpolate
    ####################
    print(xr_clim)
    print('X' * 50)

    years = np.arange(From_year, To_year)  # Need for irr
    ds_years =[]
    for year in years:
        ds_years.append(add_Rib_irr_and_interp(xr_clim, year))

    ds = xr.concat(ds_years,dim='time')

    ###############################
    # Crop and temporally crop rain
    ###############################
    # xr = xr.dropna(dim='time')
    ds = ds.where(ds.mask > 0, drop=True)  # select mask
    nb_points = ds.lon.shape[0] * ds.lat.shape[0]
    points = np.arange(nb_points).reshape(ds.mask.shape)
    ds.coords['points'] = (('lat', 'lon'), points)

    print('start ge rain')
    windows_size = 3
    xr_rain_cpc = xr_rain_cpc.sel(lat=ds.lat.values, lon=ds.lon.values, method='nearest')
    xr_rain_cpc =xr_rain_cpc.assign_coords(lat = ds.lat.values, lon=ds.lon.values)
    ds_rain = xr_rain_cpc.reindex(time=ds.time)
    ds_rain= ds_rain.rolling(time=windows_size, min_periods=1, center=True).sum() / windows_size
    ds_rain = ds_rain.assign_coords(var='precip')
    ds_rain = ds_rain.expand_dims('var')
    ds_rain = ds_rain.assign_coords(elev = ds.elev)
    ds_rain = ds_rain.assign_coords(mask = ds.mask)
    ds_rain = ds_rain.assign_coords(points = ds.points)
    # df_rain = xr_rain.to_dataframe().copy()
    # window_size = 4
    # df_rain['precip'] = df_rain['precip'].rolling(window_size, win_type='triang', min_periods=1,
    #                                               center=True).sum().div(window_size / 2)
    # df_rain['precip'] = df_rain['precip'].fillna(0)
    # ds_data2nc = xr.Dataset.from_dataframe(df_rain)

    ############################
    # Convert to sib2 variables
    ############################
    print('convert var')
    U = ds.sel(var='U')
    V = ds.sel(var='V')#.load()
    u = U['clim'].stack(s = ['lat','lon','time'])
    v = V['clim'].stack(s = ['lat','lon','time'])
    rho, theta = cart2pol(u, v)  # Convert UV in wind speed
    rho = rho.unstack('s')
    rho = rho.assign_coords(var='Vw')
    rho = rho.expand_dims('var')
    ev = Ev(ds.sel(var='Q'), ds.sel(var='P'))
    ev = ev.assign_coords(var='Ev')
    ev = ev.expand_dims('var')
    ev = ev['clim']

    ds_data2nc = xr.concat([ds['clim'].sel(var=['T','ki']), ev, rho, ds_rain],dim='var', coords='different')

    return ds_data2nc

if __name__ == '__main__':


    From_year = 2014  # TODO WARNING
    To_year = 2016


    ds_data2nc = make_data2nc_from_maihr(From_year, To_year)
    ds_data2nc.to_netcdf('/vol0/thomas.martin/framework/stamod_rib/sib2_ribeirao_pos/data2_rib.nc')
    print('done prepare data2nc')



    # rib_conf = Rib_conf()
    # luigi.build([write_all_inputs(From_year=rib_conf.From_year, To_year= rib_conf.To_year)], local_scheduler=False, workers=1) # Does notwork with 4 workers