"""
    Verification Data2 plot

"""
import xarray as xr
from sibcat_conf import Rib_conf
import matplotlib.pyplot as plt


rib_conf = Rib_conf()

ds = xr.open_mfdataset(str(rib_conf.data2_netcdf_path))

# Plot temporal series of a point in the middle of the domain
ds['clim'].isel(lat=int(len(ds.lat)/2), lon=int(len(ds.lon)/2)).plot(row='var', sharey=False)
plt.savefig('temporal_plot_all_variables_middle_domain.png')

# Spatial plot middle of the time serie
fig, axes = plt.subplots(len(ds.coords['var']),1,figsize=(7,len(ds.coords['var'])*7))
for i,v in enumerate(ds.coords['var']):
    ds['clim'].isel(time=int(len(ds.time)/2)).sel(var=v).plot(ax=axes[i])
plt.tight_layout()
plt.savefig('spatial_plot_all_variables_middle_period.png')

print('done plot')
