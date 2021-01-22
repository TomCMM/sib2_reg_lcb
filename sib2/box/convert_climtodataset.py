"""
 convert climate out dataset to be readable by Leo R

"""
import xarray as xr
import pandas as pd 
import glob
import os
from pathlib import Path


filepath= "/vol0/thomas.martin/framework/stamod_rib/out/"
files = glob.glob(filepath + '*clim*.nc')

for file in files:
    da = xr.open_dataset(file)

    print('done')

    das = xr.Dataset({
        'T':(['time','lat','lon'],da['clim'].sel(var='T').values),
        'Q':(['time','lat','lon'],da['clim'].sel(var='Q').values),
        'U':(['time','lat','lon'],da['clim'].sel(var='U').values),
        'V':(['time','lat','lon'],da['clim'].sel(var='V').values),
        'P':(['time','lat','lon'],da['clim'].sel(var='P').values)}, coords={'time':da.time.values,'lat':da.lat.values,'lon':da.lon.values})
    das.to_netcdf(str(Path(file).parent/Path(file).stem)+'_new.nc')


