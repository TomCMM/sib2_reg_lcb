"""
 Plot sibcat output
"""
import xarray as xr
import matplotlib.pyplot as plt
from pylab import *





# ####
# # Spatial plot
# ####
# ds = xr.open_dataarray(outfilepath)
# ds = ds.rename(variable='var')
# for ax, v in zip(axes, list(ds.coords['var'].values)):
#     print(v)
#     ds.sel(var=v).sel(time='2064-07-20 15:00:00').plot(ax=ax)
# plt.tight_layout()
# plt.savefig('/vol0/thomas.martin/maihr/sib2_reg_lcb/sib2/fig/sibcat_output_temporal.png')
#

################
# 6 hourly plot
################

def plot_spatial(sib2outpath, figoutpath):





    hours = [0, 6, 9, 12, 15, 18]


    ds = ds.rename(variable='var')
    ds  =  ds.groupby('time.hour').mean()

    ds = ds.sel(hour=hours)


    fig, axes = plt.subplots(len(ds.coords['var'].values), len(hours), figsize=(len(hours) * 7, len(ds.coords['var'].values) * 7))
    ds = ds.stack(var_hour=('var','hour'))

    axes = axes.flatten()
    for ax, v in zip(axes, list(ds.coords['var_hour'].values)):
        print(v)
        ax.title.set_text(ds.sel(var_hour=v).coords['long_name'].values)
        ds.sel(var_hour=v).plot(ax=ax, cmap='RdBu') #,cmap = cm.get_cmap('RdBu', 10))
    plt.tight_layout()
    plt.savefig(figoutpath)




if __name__ == '__main__':
    sib2outpath ='/vol0/thomas.martin/framework/stamod_rib/sib2_ribeirao_pos/pontual_ribeirao/out_nc/sibcat_test.nc'
    figoutpath='/vol0/thomas.martin/maihr/sib2_reg_lcb/sib2/fig/sibcat_output_spatial.png'
    plot_spatial(sib2outpath, figoutpath)