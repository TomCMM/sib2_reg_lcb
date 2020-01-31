#===============================================================================
# DESCRIPTION
# Use the parameters of the multilinear regression and the map of independant variables to generate a map of climatic variable
#===============================================================================
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from LCBnet_lib import *

from LCBnet_lib import *
from Irradiance import Irradiance_sim_obs
import glob
LCB_Irr = Irradiance_sim_obs.LCB_Irr

#===============================================================================
# Get ratio, should simplify this
#=============================================================================== 
# Note
# I am using the elevation from the DEm for ease.
# However in the future, I should put the observed altitude 
inpath = "/home/thomas/"
stanames = ['C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19']
alt = pd.read_csv(inpath+"Irradiance_rsun_lin2_lag-0.2_elev_df.csv", index_col = 0, parse_dates=True)
   
irr = LCB_Irr()
inpath_obs = '/home/thomas/PhD/obs-lcb/LCBData/obs/Irradiance/data/'
files_obs = glob.glob(inpath_obs+"*")
irr.read_obs(files_obs)
  
irr.read_sim(inpath+"Irradiance_rsun_lin2_lag-0.2_glob_df.csv")
# this should be implemented in the Grass code directly
irr.data_sim.columns = stanames
alt.columns = stanames
  
# rng = pd.date_range(irr.data_sim.index[0], irr.data_sim.index[-1],freq='1H')
# df_0 = pd.DataFrame(0,index =rng, columns=stanames)
# 
# 
# irr.data_sim = irr.data_sim.reindex(rng)
# alt = alt.reindex(rng)
# irr.data_sim = pd.merge(irr.data_sim, df_0, how='right')
# alt = pd.merge(alt, df_0, how='right')
  
# pas sur que faire le ratio avec le simuler clear sky soit 
# la meilleur forme surtout in the morning and evening transitioning
ratio = irr.ratio()



#===============================================================================
# Get map irradiance and elev
#===============================================================================
dayofyear =100
hour =15

# dayofyear = 336
# hour = 9

date = pd.to_datetime(datetime.datetime(2015, 1, 1) + datetime.timedelta(dayofyear - 1) + datetime.timedelta(hours=hour))

# date = pd.to_datetime(datetime.datetime(2013, 1, 1) + datetime.timedelta(dayofyear - 1) + datetime.timedelta(hours=hour))
irr  = np.loadtxt('/home/thomas/100_15rasterglob.txt',delimiter=',')
irr = irr* ratio.ix[date]
elev  = np.loadtxt('/home/thomas/100_15rasterelev.txt',delimiter=',')

# irr  = np.loadtxt('/home/thomas/336_9rasterglob.txt',delimiter=',')
# elev  = np.loadtxt('/home/thomas/336_9rasterelev.txt',delimiter=',')


#===============================================================================
# Get gfs data and create an array of them 
#===============================================================================

df_gfs = pd.read_csv('/home/thomas/gfs_serie.csv', index_col=0, parse_dates=True)
# df_gfs = pd.read_csv('/home/thomas/gfs_data_GFS_131202.csv', index_col=0, parse_dates=True)
data_gfs = df_gfs.ix[date]

vars = {}
vars['irr'] = irr # Only day 

plt.contourf(irr[::-1,:])
plt.colorbar()
plt.show()

vars['Alt'] =  elev

CLWMR_500mb =np.empty(irr.shape)
CLWMR_500mb.fill(data_gfs['CLWMR_500mb'])
vars['CLWMR_500mb'] = CLWMR_500mb

HGT_950mb =np.empty(irr.shape)
HGT_950mb.fill(data_gfs['HGT_950mb'])
vars['HGT_950mb'] = HGT_950mb


RH_950mb =np.empty(irr.shape)
RH_950mb.fill(data_gfs['RH_950mb'])
vars['RH_950mb'] = RH_950mb

TMP_950mb =np.empty(irr.shape)
TMP_950mb.fill(data_gfs['TMP_950mb'])
vars['TMP_950mb'] = TMP_950mb

VVEL_950mb =np.empty(irr.shape)
VVEL_950mb.fill(data_gfs['VVEL_950mb'])
vars['VVEL_950mb'] = VVEL_950mb



#===============================================================================
# Retrieve parameter and eig_pairs to reconstruct the variable map
#===============================================================================


df_params = pd.read_csv('/home/thomas/params.csv',index_col=0)


# fig, axs = plt.subplots(1,len(vars.keys()))

Y = np.zeros(irr.shape)
for i, var in enumerate(vars.keys()):
    plt.contourf(df_params[var].values * vars[var])
    plt.colorbar()
    plt.savefig('/home/thomas/'+var+'.png')
    plt.close()
    Y = Y + df_params[var].values * vars[var]

plt.show()



Y0 = np.zeros(irr.shape)
Y0 .fill(df_params.loc[0].values.item(0) )
Y = Y + Y0
    


CS = plt.contour(elev[::-1,::],colors='k')
plt.contourf(Y[::-1,:], levels = np.linspace(Y.min(), Y.max(), 100))
plt.clabel(CS, inline=1, fontsize=10)
plt.colorbar()
plt.show()


np.savetxt("/home/thomas/variable.txt", Y, delimiter=',')



