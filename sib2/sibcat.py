"""
    Run Sib2 for the Ribeirao Das Posses watershed

"""

import luigi
import datetime
import os
import signal
import pandas as pd
import xarray as xr
import numpy as np
import glob
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import Parallel, delayed

from pathlib import Path
from shutil import copy2
from time import sleep
from subprocess import Popen, PIPE
import subprocess
from pyfiglet import Figlet
from shutil import copyfile

from plot_sibcat_output import plot_spatial
from sibcat_conf import Rib_conf
# from sib2.prepare_input_sib2 import write_all_inputs # Prepare all inputs for sib2 # TODO WARNING RODOLFO export

def popen_timeout(command, timeout, cwd=None):
    p = Popen(command, stdout=PIPE, stderr=PIPE,cwd=cwd,shell=True, preexec_fn=os.setsid)
    for t in range(timeout):
        sleep(1)
        if p.poll() is not None:
            return p.communicate()
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    return False

def convert_datetime_sib2(time):
    try:
         time = datetime.datetime.strptime(time, '%y%m%d%H')
    except ValueError:
         time = time.replace('24', '23')
         time = datetime.datetime.strptime(time, "'%y%m%d%H'")
         time += datetime.timedelta(hours=1)

    return time

def write_data2_file(ds, point_nb, output_folderpath, files_sib2):
    output_folderpath = str(output_folderpath)

    tmp_dir = '/data_'+str(point_nb) + '/' # create directory for each points as the Data1 needs to be the same
    print(str(tmp_dir))

    os.makedirs(output_folderpath + tmp_dir, exist_ok=True)

    for file_sib2 in files_sib2: # copy the master sibfile in the tmp_folder
        copyfile(file_sib2, output_folderpath +tmp_dir + os.path.basename(file_sib2))

    print('UI'*100)
    print(output_folderpath + tmp_dir + 'data2')

    # ds = ds.sel(lat=lat, lon=lon)
    # xr = xr.isel(lat=10, lon=10)

    # U = xr.sel(var='U').load()
    # V = xr.sel(var='V').load()
    # rho, theta = cart2pol(U, V)  # Convert UV in wind speed
    # ev = Ev(xr.sel(var='Q'), xr.sel(var='P'))

    t = pd.to_datetime(ds.time.values) + pd.DateOffset(years=50)# shift needed for data2 dates to be in ascending order
    df_time = pd.DataFrame(t.strftime('%y%m%d%H'), index=ds.time.to_dataframe().index).astype(int)

    df_ki = ds.sel(var='ki').to_dataframe()['clim']
    df_ki = df_ki.fillna(0)

    df_ev = ds.sel(var='Ev').to_dataframe()['clim']
    df_rho = ds.sel(var='Vw').to_dataframe()['clim']
    df_T = ds.sel(var='T').to_dataframe()['clim']
    df_rain = ds.sel(var='precip').to_dataframe()['clim']
    df_9999 = pd.DataFrame([-9999.00] * len(df_time), index=df_time.index)

    df_to_file = pd.concat([df_time, df_ki, df_ev, df_T, df_rho, df_rain, df_9999, df_9999, df_9999, df_9999], axis=1,join='inner')
    df_to_file = df_to_file.fillna(method='ffill') # TODO Warning fill last mising hours of the year with

    outfilepath = output_folderpath + tmp_dir + 'data2'

    with open(outfilepath, "a") as f:# Open file to write
        print('U'*100)
        # print(df_to_file)
        # f.write("    nymd        ki        em        tm        um      prec    H        LE        Fc        u*\n")
        np.savetxt(f, X=df_to_file.values,
                   fmt=['%i', '%10.2f', '%10.2f', '%10.2f', '%10.2f', '%10.2f', '%10.2f', '%10.2f', '%10.2f', '%10.2f'])

    # tmp_dir = '/data_' + str(point) + '/'  # create directory for each points as the Data1 needs to be the same
    # filename = str(out_path) + tmp_dir + 'data2'

    header = "    nymd        ki        em        tm        um      prec    H        LE        Fc        u*\n"
    with open(outfilepath, 'r') as original:
        data = original.read()
        with open(outfilepath, 'w') as modified:
            modified.write(header + data)

def write_data2( data2nc_path):

    # points = luigi.ListParameter()
    # From_year = luigi.Parameter()
    # To_year = luigi.Parameter()
    # data2nc_path = luigi.Parameter()
    rib_conf = Rib_conf()

    framework_path = rib_conf.framework_path
    sib_folder =rib_conf.sib_folder # folder with the executable
    out_path = rib_conf.out_path
    res_path = rib_conf.res_path

    # chunck_month_by = 6  # +2 no final

    # def output(self):
    #
    #     filename = str(self.out_path /str('luigiflag__'+str(self.points[0])))
    #     return luigi.LocalTarget(filename)

    # def run(self):

        ######################
        # Write SIb2 for each year
        ######################
        # years = range(self.From_year, self.To_year)# Need for irr
        # for year in years:
        #     print(year)

    ds = xr.open_dataarray(data2nc_path)
    ds = ds.stack(points = ['lat','lon'])
    points = list(ds.points)
    # get executable
    files_sib2 = glob.glob(str(sib_folder)+'/' + '*.h')
    files_sib2.append(str(sib_folder) + '/a.out')  # sib2 executable
    files_sib2.append(str(sib_folder) + '/data1')  # data1

    print('start write')
    result = Parallel(n_jobs=int(20), prefer='threads')(delayed(write_data2_file)(ds.isel(points=i),i,  out_path, files_sib2) for i,v in enumerate(points))


    #
    # for point in points:
    #     print('writing point {}'.format(point))
    #     lat = ds.where(ds.points == point, drop=True).lat.values[0]
    #     lon = ds.where(ds.points == point, drop=True).lon.values[0]
    #     write_data2_file(lat, lon, point, ds, out_path, files_sib2)
    #
    # print('finish write')
    # # xr_rain.close()
    # ds.close()
    #
    # print('Write header')
    # for point in points:
        # tmp_dir = '/data_' + str(point) + '/'  # create directory for each points as the Data1 needs to be the same
        # filename = str(out_path) + tmp_dir + 'data2'
        #
        # header = "    nymd        ki        em        tm        um      prec    H        LE        Fc        u*\n"
        # with open(filename, 'r') as original:
        #     data = original.read()
        #     with open(filename, 'w') as modified:
        #         modified.write(header + data)
    #
    # file = open(output().path, 'w')

def write_all_data2(data2nc_path):


    # From_year = luigi.Parameter()
    # To_year = luigi.Parameter()
    #
    # def __init__(self, data2nc_path):
    #
    #     data2nc_path= luigi.Parameter()

    rib_conf = Rib_conf()
    sib_folder =rib_conf.sib_folder # folder with the executable
    out_path = rib_conf.out_path
    res_path = rib_conf.res_path

    # def run(self):

    ds = xr.open_dataset(data2nc_path)
     # todo points should only have lat/lon dimension
    # points = list(range(1836)) # Size of the extracted xr in get_xr
    points = list(range(len(ds.stack(p=['lat', 'lon']).p))) # Size of the extracted xr in get_xr
    size =  51 # len(points) / size must be integer # Todo WARNING
    points_seq = [points[i:i + size] for i in range(0, len(points), size)]

    for points in points_seq:
        write_data2(points=points, data2nc_path=data2nc_path)

def sib2_point(point, data1nc_path,idx,cols):
    """
     Run sib2 for on point

    """

    # user = luigi.Parameter()
    # data1nc_path = luigi.Parameter()

    # framework_folderpath = Path.home()
    # sib_folder = Path.home() / '.exe/'
    # output_folderpath = Path.home()  /'.out/res/'
    # input_folderpath = Path.home() / '.out/input_data/'

    rib_conf = Rib_conf()
    framework_folderpath = rib_conf.sib_folder

    sib_folder = framework_folderpath / 'exe/'
    output_folderpath = framework_folderpath  /'out/res/'
    input_folderpath = framework_folderpath/ 'out/input_data/'

    prefix_fodler = 'data_'
    # point = luigi.Parameter()
    timeout = 60*5 # WARNING TIME BEFORE STOP SIB2

    # def output(self):
    outfilename = output_folderpath / str('sib2dt_'+str(point) + '.csv')
        # return luigi.LocalTarget(str(outfilename))

    # def run(self):

    path_input_dir = input_folderpath / str(prefix_fodler + str(point))

    exe = 'a.out'

    df_data2 = pd.read_csv(str(path_input_dir /'data2'), delim_whitespace=True)
    df_data2.dropna(axis=0, how='any',inplace=True)

    print(str(point))

    if len(df_data2) ==0:
        df = pd.DataFrame()
        df.to_csv(outfilename)
    else:

        copy2(str(sib_folder/exe), str(path_input_dir /exe))


        # print(str(framework_folderpath / data1nc_path))
        # print( str(path_input_dir /'data1'))
        copy2(str(data1nc_path), str(path_input_dir /'data1'))

        # sleep(4)

        p1 = popen_timeout('chmod a+x ' +  str(path_input_dir / exe), timeout=20, cwd=path_input_dir)
        # print('Running -- ' + point)
        # p = popen_timeout(str(path_input_dir / exe), timeout=40, cwd=str(path_input_dir))
        #
        # sleep(1)
        # # subprocess.call(str(path_input_dir / exe), shell=True)
        cmd=str(path_input_dir / exe)
        # subprocess.Popen(cmd, cwd=str(path_input_dir))

        try:
            output = subprocess.check_output(cmd, cwd=str(path_input_dir), stderr=subprocess.STDOUT, timeout=timeout)


            # try:
            sib2_outfile = str(path_input_dir  /'sib2dt.dat')
            df = pd.read_csv(sib2_outfile, index_col=0, delim_whitespace=True)
            # print(df)
            # raise('STOP')
            # print('====='*100)


            df.index = pd.to_datetime(df.index, format='%y%m%d%H')
            df.index.name = 'date'

            df = df.loc[~df.index.duplicated(keep='first')]  # TODO WARNING SHOULD NOT NEED THIS
            df = df.reindex(idx)
            df = df.reindex(columns=cols)


            # print(outfilename)
            # df.to_csv(outfilename)
            os.remove(sib2_outfile)
            # print(f'done sucessfully {str(point)}')

        except Exception as e:
            print(e)
            df = pd.DataFrame(columns=cols,index=idx)
            df.index = pd.to_datetime(df.index, format='%y%m%d%H')
            # df.index.name = 'time'

        ds = df.to_xarray()
        ds = ds.to_array()
        # print(ds)
        ds = ds.rename(index='time')
        ds = ds.assign_coords(points=point)
        ds =ds.expand_dims('points')

        return ds
        # # except:
        #     df = pd.DataFrame()
        #     df.to_csv(self.output().path)
#
# class run_all_points(luigi.WrapperTask):
#     """
#     Wrapper task that call all sib2 points
#
#     """
#
#     user = luigi.Parameter()
#     data1nc_path = luigi.Parameter()
#
#     rib_conf = Rib_conf()
#     framework_folderpath = rib_conf.sib_folder
#     input_folderpath = framework_folderpath / 'out/input_data/'
#
#     # clim_input = 'clim_input.nc'
#     outfilename = 'sib2_rib.nc'
#
#     def requires(self):
#         tasks = {}
#
#         folders = glob.glob(str(self.input_folderpath)+'/data*')
#         points = [ folder.split('_')[-1] for folder in folders]
#
#         # xr_clim_input = xr.open_dataset(str(self.input_folderpath / self.clim_input))
#         # xr_clim_input = xr_clim_input.sel(lat=xr_clim_input.lat.values,lon=xr_clim_input.lon.values)
#         # points = np.array(xr_clim_input.points.values).flatten().tolist()
#         # points = list(range(1836)) # Size of the extracted xr in get_xr
#
#         for point in points:
#             tasks[str(point)] = sib2_point(point=str(point),user=self.user, data1nc_path=self.data1nc_path)
#         return tasks
#
#     def output(self):
#         return self.input()

def SibCat(data1nc_path,data2nc_path,outpath, nb_process=50 ):
    """
    Master class that give final output from all points
    """
    #
    # user = luigi.Parameter()
    # data1nc_path = luigi.Parameter()
    # data2nc_path = luigi.Parameter()

    f = Figlet(font='standard')
    print(f.renderText('SibCat'))

    nb_var_sib = 43

    # #spring
    rib_conf = Rib_conf()
    framework_folderpath = rib_conf.sib_folder

    output_folderpath = framework_folderpath
    clim_input = 'clim_input.nc'

    framework_folderpath = rib_conf.sib_folder
    input_folderpath = framework_folderpath / 'out/input_data/'

    folders = glob.glob(str(input_folderpath) + '/data*')
    # points = [folder.split('_')[-1] for folder in folders]

    # xr_clim_input = xr.open_dataset(str(self.input_folderpath / self.clim_input))
    # xr_clim_input = xr_clim_input.sel(lat=xr_clim_input.lat.values,lon=xr_clim_input.lon.values)
    # points = np.array(xr_clim_input.points.values).flatten().tolist()
    # points = list(range(1836)) # Size of the extracted xr in get_xr

    ############
    # Run all points
    ############
    # # for point in points:

    #######################
    # Export file netcdf
    #######################
    # raise
    rib_conf = Rib_conf()

    ds_data2 = xr.open_dataarray(str(data2nc_path)) # just to get the coords
    # ds_data2 = ds_data2.where(ds_data2.mask > 0, drop=True)  # select mask
    # nb_points = ds_data2.lon.shape[0] * ds_data2.lat.shape[0]
    # points = np.arange(nb_points).reshape(ds_data2.mask.shape)
    # ds_data2.coords['points'] = (('lat', 'lon'), points) # TODO SHould not be needed
    # ds_data2 = ds_data2.stack(points=('lat','lon'))
    ds_data2 = ds_data2.drop('points') # TODO put in prepare data
    ds_data2 = ds_data2.stack(points = ('lat','lon'))

    idx = pd.DatetimeIndex(ds_data2.time.values) + pd.offsets.DateOffset(years=50)
    cols=['date','Tm','em','um','Ki','alb','Ldwn','Lupw','Rn_C','H_C','LE_C',
          'G_C','J_C','Fc_C','Rsc_C','An_C','u*_C','Td','W1_C','W2_C','W3_C',
          'W4_C','W5_C','W6_C','W7_C','W8_C','W9_C','W10_C','gc','Evpt','Trans',
          'Esoil','Einterc','Prec','Rss','Rs','Runoff','PARidir','PARidif','albPARdir','albPARdif','Tc','Ta','PPB']

    dfs = Parallel(n_jobs=int(nb_process), prefer='processes')(delayed(sib2_point)(point,data1nc_path,idx,cols) for point in range(len(ds_data2.points)))
        # sib2_point(point=str(point), data1nc_path=data1nc_path)

    # inputs = self.input()


    # print('Write ' + str(self.output().path))


    # points = list(points.flatten())
    # points = list(range(1836))
    #
    # From_year = pd.Timestamp(ds_data2.time[0].values).year # TODO WARNING
    # To_year = pd.Timestamp(ds_data2.time[-1].values).year
    #
    # years = np.arange(From_year, To_year)# Need for irr
    # years = years + 50
    #
    # for year in years:
    #     print('='*100)
    #     print(year)
    #     arr = []
    #
    #     idx = valid_point.index
    #     cols = valid_point.columns
    #     idx_year = valid_point.loc[str(year),:].index

    # for point in points:
    #
    # def read_csv(df):
    #     # output_folderpath = framework_folderpath / 'out/res/'
    #     #
    #     # outfilename = output_folderpath / str('sib2dt_' + point + '.csv')
    #     #
    #     # print(point)
    #     # df = pd.read_csv(str(outfilename),index_col=0, parse_dates=True)
    #     df = df.loc[~df.index.duplicated(keep='first')] # TODO WARNING SHOULD NOT NEED THIS
    #     df = df.reindex(idx)
    #     df = df.reindex(columns=cols)
    #     # arr.append(df.values)
    #
    #     return df.values
    #
    # with tqdm(desc='Sib2-output', total=len(points)):
    #     arr = Parallel(n_jobs=int(40), prefer='threads')(delayed(read_csv)(df) for df in dfs)

    # for point in points:


    ds = xr.concat([df for df in dfs if df is not None],dim='points') # TODO i don't know why their is some None in the list
    print('done concat')
    # ds =ds.chunk({'variable':1,'points':200})
    # print('chunk')
    # print(ds.sel(variable='Ta').isel(time=5000).mean())
    # print('reindex')
    ds = ds.reindex(points=range(len(ds_data2.points))) # TODO Points need to have only 2 dimensions lat/loon
    #
    # print(ds.sel(variable='Ta').isel(time=5000).mean())
    # print(ds)
    print('assign coord')
    ds = ds.assign_coords(points=ds_data2.points)
    # print(ds.sel(variable='Ta').isel(time=5000).mean())
    # print(ds)
    # ds = ds.assign_coords(points = ds_data2.points)
    print(ds)
    print('unstack')
    ds = ds.unstack('points')
    # print(ds.sel(variable='Ta').isel(time=5000).mean())
    # print('done')
    # print(ds)
    # xarr = np.dstack(dfs)
    #
    # xarr_reshape = xarr.reshape(len(idx),len(vars),len(ds_data2.lat), len(ds_data2.lon))
    # ds = xr.DataArray(xarr_reshape, dims=['time','var', 'lat','lon'])
    # ds = ds.assign_coords(lat=ds_data2.lat.values)
    # ds = ds.assign_coords(lon=ds_data2.lon.values)
    # ds = ds.assign_coords(time=idx)
    # ds = ds.assign_coords(var=vars)
    # # ds = ds.assign_coords(points=points)
    # ds.coords['points'] = (('lat', 'lon'), ds_data2['points'].values)
    # # ds.coords['elev'] = (('lat', 'lon'), ds_clim_input['elev'].values)
    #
    # ds = ds.to_dataset(dim='var') # convert to dataset t be compatible with R
    # outpath = os.path.dirname(self.output().path) +'/sib2_rib' + self.data1nc_path + '_' + str(year) + '.nc'




    ds = xr.open_dataarray(sib2outpath)

    coords_units = {
        'Tm': 'temperatura do ar observada (K)',
         'em': 'pressão de vapor d’água observada (hPa)',
         'um': 'velocidade horizontal do vento observado (m.s-1)',
         'Ki':'irradiância de onda curta incidente observada (W.m-2)',
         'alb':'albedo',
         'Ldwn':'irradiância de onda longa incidente (W.m-2)',
         'Lupw': 'irradiância de onda longa emergente (W.m-2)',
         'Rn_C':'saldo de radiação calculado (W.m-2)',
         'H_C': 'fluxo de calor sensivel calculado (W.m-2)',
         'LE_C':'fluxo de calor latente calculado (W.m-2)',
         'G_C':'fluxo de calor no solo calculado (W.m-2)',
         'J_C':'fluxo de energia armazenado na coluna de ar do dossel (W.m-2)',
         'Fc_C':'fluxo total de CO2 (μmolCO2.m-2.s-1)',
         'Rsc_C': 'efluxo de Co2 do solo  (μmolCO2.m-2.s-1)',
         'An_C': 'Assinilacao liquida de Co2 (μmolCO2.m-2.s-1)',
         'u*_C': 'velocidade de atrito (m.s-1)',
         'Td':'Temperatura do solo (C)',
         'W1_C':'grau de saturacao da umidade do solo 1 camada (m3.m-3)',
         'W2_C':'grau de saturacao da umidade do solo 1 camada (m3.m-3)',
         'W3_C':'grau de saturacao da umidade do solo 1 camada (m3.m-3)',
         'W4_C':'grau de saturacao da umidade do solo 1 camada (m3.m-3)',
         'W5_C':'grau de saturacao da umidade do solo 1 camada (m3.m-3)',
         'W6_C':'grau de saturacao da umidade do solo 1 camada (m3.m-3)',
         'W7_C':'grau de saturacao da umidade do solo 1 camada (m3.m-3)',
         'W8_C':'grau de saturacao da umidade do solo 1 camada (m3.m-3)',
         'W9_C':'grau de saturacao da umidade do solo 1 camada (m3.m-3)',
         'W10_C':'grau de saturacao da umidade do solo 1 camada (m3.m-3)',
         'gc': 'Conductancia do dossel (mm.s-1)',
         'Evpt':'Evapotranspiracao (mm)',
         'Trans': 'Transpiracao (mm)',
         'Esoil': 'Evaporacao do solo (mm)',
         'Einterc': 'Evaporacao por interceptacao (mm)',
         'Prec': 'Precipitacao (mm)',
         'Rss': 'Escoamento por drenagem vertical para aquifero (mm)',
         'Rs': 'Escoamento de superficie (mm)',
         'Runoff': 'Escoamento total = croff + qng (mm)',
         'PARidir':'Irradiancia PAR incidente direta (W.m-2)',
         'PARidif':'Irradiancia PAR incidente difusa (W.m-2)',
         'albPARdir':'Albedo PAR de radiacao direta (adimensional)',
         'albPARdif':'Albedo PAR de radiacao difusa (adimensional)',
         'Tc': 'temperatura do dossel',
         'Ta': 'temperatura do ar',
         'PPB':'prod primária bruta'}

    ds = ds.sel(variable = list(coords_units.keys()))
    ds = ds.assign_coords(long_name=('variable',list(coords_units.values())))


    ds = ds.isel(variable=slice(1,None)) # TODO to remove, Remove the date


    print(outpath)
    print(ds)
    ds.to_netcdf(outpath)
    #
    # file = open(self.output().path, 'w')
    # # remove all .csv
    # res = self.framework_folderpath /'out/res/'
    # res = res.glob('*.csv')
    #
    # for file in list(res):
    #     os.remove(str(file))

if __name__ =='__main__':

    rib_conf = Rib_conf()

    write_data2(data2nc_path=rib_conf.data2nc_path)

    outfilepath ='/vol0/thomas.martin/framework/stamod_rib/sib2_ribeirao_pos/pontual_ribeirao/out_nc/sibcat_test.nc'

    SibCat(data1nc_path= rib_conf.data1nc_path,
           data2nc_path= rib_conf.data2nc_path,
           nb_process=80,
           outpath=outfilepath)


    sib2outpath ='/vol0/thomas.martin/framework/stamod_rib/sib2_ribeirao_pos/pontual_ribeirao/out_nc/sibcat_test.nc'
    figoutpath='/vol0/thomas.martin/maihr/sib2_reg_lcb/sib2/fig/sibcat_output_spatial.png'
    plot_spatial(outfilepath, figoutpath)





