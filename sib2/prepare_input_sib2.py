"""

    SIB2: Land surface model

"""

import pandas as pd
import glob

from shutil import copyfile
import os
import xarray
import numpy as np
import luigi
import dask

from mailib.toolbox.geo import cart2pol
from mailib.toolbox.meteo import Ev
from maihr_config import Rib_conf #, Maihr_Conf # To export to rodolfo
# from downscaling_ribeirao.statmod_articleII import predict_stamod_rib, predict_csf_posses

from dateutil.relativedelta import relativedelta


from multiprocessing.pool import ThreadPool
dask.config.set(pool=ThreadPool(20))

def write_data2(lat, lon, point, xr, xrain, output_folderpath, files_sib2):
    output_folderpath = str(output_folderpath)

    tmp_dir = '/data_'+str(point) + '/' # create directory for each points as the Data1 needs to be the same
    print(str(tmp_dir))


    os.makedirs(output_folderpath + tmp_dir, exist_ok=True)

    for file_sib2 in files_sib2: # copy the master sibfile in the tmp_folder
        copyfile(file_sib2, output_folderpath +tmp_dir + os.path.basename(file_sib2))

    print('UI'*100)

    print(output_folderpath + tmp_dir + 'data2')

    xr = xr.sel(lat=lat, lon=lon)
    # xr = xr.isel(lat=10, lon=10)


    U = xr.sel(var='U').load()
    V = xr.sel(var='V').load()
    rho, theta = cart2pol(U, V)  # Convert UV in wind speed
    ev = Ev(xr.sel(var='Q'), xr.sel(var='P'))


    t = pd.to_datetime(xr.time.values) + pd.DateOffset(years=50)# shift needed for data2 dates to be in ascending order
    df_time = pd.DataFrame(t.strftime('%y%m%d%H'), index=xr.time.to_dataframe().index).astype(int)


    df_ki = xr.sel(var='ki').to_dataframe()['clim']
    df_ki = df_ki.fillna(0)

    df_ev = ev.to_dataframe()['clim']
    df_rho = rho.to_dataframe()['clim']
    df_T = xr.sel(var='T').to_dataframe()['clim']
    df_rain = xrain['precip'].to_dataframe()['precip']
    df_9999 = pd.DataFrame([-9999.00] * len(df_time), index=df_time.index)


    df_to_file = pd.concat([df_time, df_ki, df_ev, df_T, df_rho, df_rain, df_9999, df_9999, df_9999, df_9999], axis=1,join='inner')
    df_to_file = df_to_file.fillna(method='ffill') # TODO Warning fill last mising hours of the year with

    outfilepath = output_folderpath + tmp_dir + 'data2'

    # print(lat)
    # print(lon)
    # print(xr.sel(var='T').values)
    # print(df_to_file)
    # raise Exception('stop!')

    with open(outfilepath, "a") as f:# Open file to write
        print('U'*100)
        print(df_to_file)
        # f.write("    nymd        ki        em        tm        um      prec    H        LE        Fc        u*\n")
        np.savetxt(f, X=df_to_file.values,
                   fmt=['%i', '%10.2f', '%10.2f', '%10.2f', '%10.2f', '%10.2f', '%10.2f', '%10.2f', '%10.2f', '%10.2f'])


    #     for date in xr.time:
    #         if float(xr.sel(var='T',time=date).values.tolist()) == float(273.15): # todo this is to be sure to not have bad lines Should be improvesd
    #             raise ValueError('The temperature is 0')
    #
    #         f.write(str(pd.to_datetime(date.values).strftime('%y%m%d%H')) + str("{0:.2f}".format(xr.sel(var='ki',time=date).values)).rjust(10, ' ')
    #                 + str("{0:.2f}".format(ev.sel(time=date).values)).rjust(10, ' ') + str("{0:.2f}".format(xr.sel(var='T',time=date).values)).rjust(10, ' ')
    #                 + str("{0:.2f}".format(rho.sel(time=date).values)).rjust(10,' ') + str("{0:.2f}".format(xrain['precip'].sel(time=date).values)).rjust(10, ' ')
    #                 +"  -9999.00  -9999.00  -9999.00  -9999.00\n")
    #

def irr_df_to_xr(out_irr_nc):
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
        xrs.append(xarray.DataArray(df_irr.values[:, i].reshape(shape_ribeirao), coords={'lat': lat, 'lon': lon},dims=['lat', 'lon']))
    xr_irr = xarray.concat(xrs, dim='time')
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

def add_irr_and_interp(xr_clim, year):

    conf = Maihr_Conf()
    framework_path  = conf.framework_path

    out_irr_nc = str(framework_path/'database/predictors/out/irr/xr_irr_rib.nc')
    if os.path.exists(out_irr_nc):
        xr_irr = xarray.open_dataset(out_irr_nc)
    else:
        xr_irr = irr_df_to_xr(out_irr_nc) # TO make xarray irr

    print('reindex hourly')
    dates_hourly = pd.date_range(str(pd.to_datetime(xr_irr.time.values).year[0]),str(pd.to_datetime(xr_irr.time.values).year[0]+1), freq='H', closed='left')
    xr_irr = xr_irr.reindex(time=dates_hourly) # It has a duplicated value I dont know why
    xr_irr.coords['time'] = pd.DatetimeIndex(xr_irr.time.values)  + pd.DateOffset(years=year-2000)
    xr_irr = xr_irr.interp(lat=xr_clim.lat, lon=xr_clim.lon, method='nearest')

    # Predict csf iir
    model_path = str(framework_path) + "/model/out/statmod/irr_rib_model/csf_model_v9.h5"
    df_pred_csf = predict_csf_posses(model_path)

    df_pred_csf = df_pred_csf.reindex(pd.date_range(df_pred_csf.index[0], df_pred_csf.index[-1], freq='H')) # Reindex hurly
    df_pred_csf = df_pred_csf.interpolate('slinear')
    df_pred_csf = df_pred_csf.loc[pd.to_datetime(xr_irr.time.values)].values[:, np.newaxis]

    xr_irr = xr_irr * df_pred_csf
    # Drop duplicates
    _, index = np.unique(xr_irr['time'], return_index=True)
    xr_irr = xr_irr.isel(time=index)


    dates_hourly = pd.date_range(str(pd.to_datetime(xr_irr.time.values).year[0]),str(pd.to_datetime(xr_irr.time.values).year[0]+1), freq='H', closed='left')

    # Drop duplicates
    _, index = np.unique(xr_clim['time'], return_index=True)
    xr_clim = xr_clim.isel(time=index)
    xr_clim = xr_clim.interp(time=dates_hourly, method='cubic')
    xr_clim = xr_clim.reindex(time=dates_hourly, method='nearest')

    xr_irr.coords['mask'] = (('lat', 'lon'), xr_clim['mask'].values)
    xr_irr = xr_irr.assign_coords(var='ki')

    T = xr_clim['clim'].sel(var='T') + 273.15 # TODO CANNOT assign in dask array
    T = T.to_dataset(name='clim')
    xr_clim = xr_clim.drop("T", dim='var')

    xr_clim =  xarray.concat([xr_clim,T], dim='var')
    print('finished get clim')


    xr_clim.coords['elev'] =(('lat', 'lon'), xr_irr['elev'].values)
    xr = xarray.concat([xr_clim, xr_irr], dim='var')



    return xr

class write_input_sib2(luigi.Task):

    points = luigi.ListParameter()
    From_year = luigi.Parameter()
    To_year = luigi.Parameter()

    rib_conf = Rib_conf()
    conf = Maihr_Conf()

    framework_path = conf.framework_path
    sib_folder =rib_conf.sib_folder # folder with the executable
    out_path = rib_conf.out_path
    res_path = rib_conf.res_path

    chunck_month_by = 6  # +2 no final

    def requires(self):
        tasks = {}

        # article
        project_name = 'stamod_rib'
        version_name = 'articleII'
        #
        # years = range(self.From_year, self.To_year)
        # for year in years:

        tasks = predict_stamod_rib(project_name=project_name, version_name=version_name,
                                             From=str(self.From_year), To=str(self.To_year), decrease_res_by=4,temporal_model_kind = 'dnn',) # Todo warning evennot set correclty the system continue to work

        return tasks

    def output(self):

        filename = str(self.out_path /str('luigiflag__'+str(self.points[0])))
        return luigi.LocalTarget(filename)

    def run(self):


        ######################
        # Climate Input
        ######################
        xr_clim_paths = glob.glob(os.path.dirname(self.input().path)+'/clim_*.nc')
        xr_clim = xarray.open_mfdataset(xr_clim_paths)
        xr_clim = xr_clim.reindex(time=sorted(xr_clim.time.values))
        xr_clim = xr_clim.load() # WARNING this could blow up the memmory
        xr_clim = xr_clim.chunk(chunks={'lat':10,'lon':10})
        # time.sleep(int(np.random.randint(120)))

        ######################
        # Rain
        ######################
        file_cpc = self.framework_path / 'stamod_rib/data/precip_1979a2017_CPC_AS.nc'
        xr_rain_cpc = xarray.open_dataarray(str(file_cpc))
        xr_rain_cpc.coords['lon'].values = (xr_rain_cpc.coords['lon'].values + 180) % 360 - 180
        xr_rain_cpc = xr_rain_cpc.sel(lat=xr_clim.lat.values[0], lon=xr_clim.lon.values[0], method='nearest')
        # xr_rain = xr_rain.sel(time=slice(dates[0], dates[-1]))

        ######################
        # Write SIb2 for each year
        ######################
        years = range(self.From_year, self.To_year)# Need for irr
        for year in years:
            print(year)
            ###################
            # Interpolate
            ####################
            print(xr_clim)
            print('X'*50)
            xr = add_irr_and_interp(xr_clim, year)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # xr = xr.dropna(dim='time')
            xr = xr.where(xr.mask > 0, drop=True)  # select mask
            nb_points = xr.lon.shape[0] * xr.lat.shape[0]
            points = np.arange(nb_points).reshape(xr.mask.shape)
            xr.coords['points'] = (('lat', 'lon'), points)

            print('start ge rain')
            xr_rain = xr_rain_cpc.reindex(time=xr.time)
            df_rain = xr_rain.to_dataframe().copy()
            window_size = 4
            df_rain['precip'] = df_rain['precip'].rolling(window_size, win_type='triang', min_periods=1,
                                                          center=True).sum().div(window_size / 2)
            df_rain['precip'] = df_rain['precip'].fillna(0)
            xr_rain = xarray.Dataset.from_dataframe(df_rain)

            # get executable
            files_sib2 = glob.glob(str(self.sib_folder)+'/' + '*.h')
            files_sib2.append(str(self.sib_folder) + '/a.out')  # sib2 executable
            files_sib2.append(str(self.sib_folder) + '/data1')  # data1

            print('start write')

            for point in self.points:
                print('writing point {}'.format(point))
                lat = xr.where(xr.points == point, drop=True).lat.values[0]
                lon = xr.where(xr.points == point, drop=True).lon.values[0]
                write_data2(lat, lon, point, xr['clim'],  xr_rain, self.out_path, files_sib2)

            print('finish write')
            xr_rain.close()
            xr.close()

        print('Write header')
        for point in self.points:
            tmp_dir = '/data_' + str(point) + '/'  # create directory for each points as the Data1 needs to be the same
            filename = str(self.out_path) + tmp_dir + 'data2'


            header = "    nymd        ki        em        tm        um      prec    H        LE        Fc        u*\n"
            with open(filename, 'r') as original:
                data = original.read()
                with open(filename, 'w') as modified:
                    modified.write(header + data)


        file = open(self.output().path, 'w')

class write_all_inputs(luigi.WrapperTask):


    From_year = luigi.Parameter()
    To_year = luigi.Parameter()

    rib_conf = Rib_conf()
    sib_folder =rib_conf.sib_folder # folder with the executable
    out_path = rib_conf.out_path
    res_path = rib_conf.res_path

    def requires(self):
        points = list(range(1836)) # Size of the extracted xr in get_xarray
        size =  51 # len(points) / size must be integer # Todo WARNING
        points_seq = [points[i:i + size] for i in range(0, len(points), size)]

        tasks = []
        for points in points_seq:
            tasks.append(write_input_sib2(points=points, From_year=self.From_year, To_year=self.To_year))

        return tasks

if __name__ == '__main__':

    # global framework_path

    # notebook
    # framework_path = Path("/home/thomas/pro/research/framework/")

    # # Spring
    # framework_path = Path("/vol0/thomas.martin/framework/") # TODO WARNING

    rib_conf = Rib_conf()
    luigi.build([write_all_inputs(From_year=rib_conf.From_year, To_year= rib_conf.To_year)], local_scheduler=False, workers=1) # Does notwork with 4 workers


