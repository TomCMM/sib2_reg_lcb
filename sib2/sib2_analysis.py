"""
Perform PCA analysis of ribeirao SIb2 .nc

"""

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dask

from eofs.xarray import Eof

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

from multiprocessing.pool import ThreadPool
dask.config.set(pool=ThreadPool(5))

from sib2.analysis_spring_carla import get_shp_masks

def plot_pca_analysis(ds, fig_output_path, title=''):
    print(title)
    # var = "T"
    print('done load')

    nbpcs = 3

    solver = Eof(ds.dropna(dim="time", how="all"))
    pcas = solver.pcs(npcs=nbpcs, pcscaling=1)
    eofs = solver.eofs(neofs=nbpcs, eofscaling=1)

    fig, axes = plt.subplots(5, 2, figsize=(20, 20))
    fig.suptitle(title, fontsize=12)

    pcas.plot.line(ax=axes[0, 0], x='time')

    pcas.resample(time='M').mean('time').plot.line(ax=axes[1, 0], x='time')
    axes[1,0].set_title('Monthly mean')

    pcas.resample(time='Y').mean('time').plot.line(ax=axes[2, 0], x='time')
    axes[2,0].set_title('Annual mean')

    pcas.groupby('time.month').mean('time').plot.line(ax=axes[3, 0], x='month')
    axes[3,0].set_title('By Month')

    pcas.groupby('time.hour').mean('time').plot.line(ax=axes[4, 0], x='hour')
    axes[4,0].set_title('By Hour')

    for pc in range(nbpcs):
        # eofs.isel(mode=pc).plot(ax=axes[pc, 1])
        eofs.to_dataframe().unstack().T.loc[:,pc].plot.bar(ax=axes[pc, 1])

    solver.varianceFraction().isel(mode=slice(0, nbpcs)).plot(ax=axes[3, 1])

    plt.tight_layout()
    plt.tight_layout()
    fig.suptitle(title)
    plt.savefig(fig_output_path + title + '.pdf', bbox_inches='tight')


if __name__ == '__main__':

    fig_output_path = "/vol0/thomas.martin/framework/stamod_rib/sib2_ribeirao_pos/pontual_ribeirao/fig/"
    # ds = xr.open_dataset("/home/thomas/pro/research/framework/stamod_rib/out/clim_2014.nc")
    ds = xr.open_mfdataset('/vol0/thomas.martin/framework/stamod_rib/sib2_ribeirao_pos/pontual_ribeirao/out_nc/*.nc')

    ds = ds.chunk(1500)

    _, index = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index)
    ds = ds.reindex({'time': pd.DatetimeIndex(ds.time.values)})
    ds = ds.transpose('time', 'lat', 'lon')

    ds_springs = []
    for var in list(ds.data_vars):
        ds = ds[var].load()
        plot_pca_analysis(ds, var, fig_output_path)

        ds[var]


