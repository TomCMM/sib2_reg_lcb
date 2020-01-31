"""
    Run Sib2 for the Ribeirao Das Posses watershed

"""

import luigi
import datetime
import os
import signal
import pandas as pd
import xarray
import numpy as np
import glob

from pathlib import Path
from shutil import copy2
from time import sleep
from subprocess import Popen, PIPE
import subprocess
from pyfiglet import Figlet

from maihr_config import Rib_conf
from sib2.prepare_input_sib2 import write_all_inputs # Prepare all inputs for sib2

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

class sib2_point(luigi.Task):
    """
     Run sib2 for on point

    """

    user = luigi.Parameter()
    data1_name = luigi.Parameter()

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
    point = luigi.Parameter()
    timeout = 60*5 # WARNING TIME BEFORE STOP SIB2

    def output(self):
        outfilename = self.output_folderpath / str('sib2dt_'+self.point + '.csv')
        return luigi.LocalTarget(str(outfilename))

    def run(self):

        path_input_dir = self.input_folderpath / str(self.prefix_fodler + self.point)

        exe = 'a.out'
        data1_name = self.data1_name


        df_data2 = pd.read_csv(str(path_input_dir /'data2'), delim_whitespace=True)
        df_data2.dropna(axis=0, how='any',inplace=True)


        if len(df_data2) ==0:
            df = pd.DataFrame()
            df.to_csv(self.output().path)
        else:

            copy2(str(self.sib_folder/exe), str(path_input_dir /exe))


            print(str(self.framework_folderpath / data1_name))
            print( str(path_input_dir /'data1'))
            copy2(str(self.framework_folderpath / data1_name), str(path_input_dir /'data1'))

            sleep(4)

            p1 = popen_timeout('chmod a+x ' +  str(path_input_dir / exe), timeout=20, cwd=path_input_dir)
            print('Running -- ' + self.point)
            # p = popen_timeout(str(path_input_dir / exe), timeout=40, cwd=str(path_input_dir))
            #
            # sleep(1)
            # # subprocess.call(str(path_input_dir / exe), shell=True)
            cmd=str(path_input_dir / exe)
            # subprocess.Popen(cmd, cwd=str(path_input_dir))

            output = subprocess.check_output(cmd, cwd=str(path_input_dir), stderr=subprocess.STDOUT, timeout=self.timeout)
                # , shell=False,
                #         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # output, error = sp.communicate()
            # sleep(self.timeout)



            # timeout = 40
            # for t in range(timeout):
            #     # print(t)
            #     # print(p.poll())
            #     sleep(1)
            #     if p.poll() is not None:
            #         p.communicate()
            # os.killpg(os.getpgid(p.pid), signal.SIGTERM)


            # try:
            sib2_outfile = str(path_input_dir  /'sib2dt.dat')
            df = pd.read_csv(sib2_outfile, index_col=0, delim_whitespace=True)
            # print(df)
            # raise('STOP')
            # print('====='*100)


            df.index = pd.to_datetime(df.index, format='%y%m%d%H')
            df.index.name = 'date'
            df.to_csv(self.output().path)
            os.remove(sib2_outfile)
        # # except:
        #     df = pd.DataFrame()
        #     df.to_csv(self.output().path)

class run_all_points(luigi.WrapperTask):
    """
    Wrapper task that call all sib2 points

    """

    user = luigi.Parameter()
    data1_name = luigi.Parameter()

    rib_conf = Rib_conf()
    framework_folderpath = rib_conf.sib_folder
    input_folderpath = framework_folderpath / 'out/input_data/'

    # clim_input = 'clim_input.nc'
    outfilename = 'sib2_rib.nc'

    def requires(self):
        tasks = {}

        folders = glob.glob(str(self.input_folderpath)+'/data*')
        points = [ folder.split('_')[-1] for folder in folders]

        # xr_clim_input = xarray.open_dataset(str(self.input_folderpath / self.clim_input))
        # xr_clim_input = xr_clim_input.sel(lat=xr_clim_input.lat.values,lon=xr_clim_input.lon.values)
        # points = np.array(xr_clim_input.points.values).flatten().tolist()
        # points = list(range(1836)) # Size of the extracted xr in get_xarray

        for point in points:
            tasks[str(point)] = sib2_point(point=str(point),user=self.user, data1_name=self.data1_name)
        return tasks

    def output(self):
        return self.input()

class sib2_ribeirao(luigi.Task):
    """
    Master class that give final output from all points
    """

    user = luigi.Parameter()
    data1_name = luigi.Parameter()

    nb_var_sib = 43

    # config = Maihr_Conf()

    # #spring
    rib_conf = Rib_conf()
    framework_folderpath = rib_conf.sib_folder

    output_folderpath = framework_folderpath

    clim_input = 'clim_input.nc'

    def requires(self):
        return run_all_points(data1_name = self.data1_name, user=self.user)

    def output(self):
        print(self.data1_name)
        outfilename = self.rib_conf.out_nc_path

        if not os.path.exists(str(outfilename)):
            os.makedirs(str(outfilename))

        return luigi.LocalTarget(str(outfilename  / str('sib2_rib' + self.data1_name + '.dummy')))

    def run(self):

        rib_conf = Rib_conf()

        xr_clim_input = xarray.open_dataarray(str(rib_conf.clim_input / 'clim_2014-01-01 00:00:00.nc')) # just to get the coords
        xr_clim_input = xr_clim_input.where(xr_clim_input.mask > 0, drop=True)  # select mask
        nb_points = xr_clim_input.lon.shape[0] * xr_clim_input.lat.shape[0]
        points = np.arange(nb_points).reshape(xr_clim_input.mask.shape)
        xr_clim_input.coords['points'] = (('lat', 'lon'), points)

        inputs = self.input()


        print('Write ' + str(self.output().path))


        # points = list(points.flatten())
        points = list(range(1836))

        years = np.arange(rib_conf.From_year, rib_conf.To_year)# Need for irr
        years = years+50
        
        for year in years:
            print('='*100)
            print(year)
            arr = []
            valid_point = pd.read_csv(inputs['975'].path, index_col=0, parse_dates=True)  # TODO dangerous


            idx = valid_point.index
            cols = valid_point.columns
            idx_year = valid_point.loc[str(year),:].index


            for point in points:
                print(point)
                df = pd.read_csv(inputs[str(point)].path,index_col=0, parse_dates=True)

                df = df.loc[~df.index.duplicated(keep='first')] # TODO WARNING SHOULD NOT NEED THIS

                df = df.reindex(idx_year)
                # df = df.loc[idx_year,:]
                df = df.reindex(columns=cols)
                vars = df.columns
                print(df.shape)
                arr.append(df.values)

            xarr = np.dstack(arr)

            xarr_reshape = xarr.reshape(len(idx_year),len(vars),len(xr_clim_input.lat), len(xr_clim_input.lon))
            xr = xarray.DataArray(xarr_reshape, dims=['time','var', 'lat','lon'])
            xr = xr.assign_coords(lat=xr_clim_input.lat.values)
            xr = xr.assign_coords(lon=xr_clim_input.lon.values)
            xr = xr.assign_coords(time=idx_year.values)
            xr = xr.assign_coords(var=vars)
            # xr = xr.assign_coords(points=points)
            xr.coords['points'] = (('lat', 'lon'), xr_clim_input['points'].values)
            # xr.coords['elev'] = (('lat', 'lon'), xr_clim_input['elev'].values)

            ds = xr.to_dataset(dim='var') # convert to dataset t be compatible with R
            outpath = os.path.dirname(self.output().path)+'/sib2_rib'+self.data1_name+'_'+str(year)+'.nc'
            print(outpath)
            print(ds)
            ds.to_netcdf(outpath)

        file = open(self.output().path, 'w')
        # remove all .csv
        res = self.framework_folderpath /'out/res/'
        res = res.glob('*.csv')

        for file in list(res):
            os.remove(str(file))

if __name__ =='__main__':
    f = Figlet(font='standard')
    print('==' * 50)
    print(f.renderText('LCB - Sib2 - Ribeirao'))
    print('==' * 50)

    # User input
    rib_conf = Rib_conf()
    data1_name = str(rib_conf.sib_folder /'data1')

    print("The name of the data1 used is :" + data1_name)
    print('The output file name will be :' 'sib2_rib_'+data1_name+'.nc' )

    # luigi.build([write_all_inputs(From_year=rib_conf.From_year, To_year= rib_conf.To_year)], local_scheduler=True, workers=1)
    luigi.build([sib2_ribeirao(data1_name=data1_name, user=Path.home().stem)], workers=40, local_scheduler=False)
    # run_ribeirao()
