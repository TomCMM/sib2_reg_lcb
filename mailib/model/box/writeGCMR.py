#!/usr/bin/python
#DESCRITPION
#       1) Import a Grib file 
#	2) write a table to imported in R

# TIPS
## Print an inventory of the file
#grbs.seek(0)
#for grb in grbs:
#        grb


import pygrib # module to import Grib file 
import pandas# to write as a data frame
import numpy as np
# ====== Input user
InPath='/data1/arps/dataI/gfs/12-17_12_2013/'
OutPath='/home/thomas/PhD/supmod/lmr_R/obs/gfs-inputR/2013-12-saopaulo/'
Outfilename='gfs-data'

month=12
day=11# start at 01 and end at 30 or 31
hour=['00','06','12','18']
fhour=['03',"06"]#forecast hour
ForJ=6#for X jour -> min 1 jour

LatS=23.5#S
LonW=46.5#W



# ====== Open file 
LatN=-LatS#N
LonE=(360-LonW)#E

for j in range(0,ForJ):
	for h in hour:
		for fh in fhour:
			filename=str(month)+str(day+j).zfill(2)+"-"+"gfs.t"+str(h)+"z.pgrb2f"+fh
			date=str(month)+str(day+j).zfill(2)+str(int(h)+int(fh)).zfill(2)
			grbs = pygrib.open(InPath+filename) # open the grib file
			print("Open "+filename)
			T2m,lats,lons=grbs[223].data(lat1=LatN,lat2=LatN,lon1=LonE,lon2=LonE)# Temperature in K
			H2m=grbs[225].data(lat1=LatN,lat2=LatN,lon1=LonE,lon2=LonE)[0] #Relative humidity 2m [%]
			Kis=grbs[274].data(lat1=LatN,lat2=LatN,lon1=LonE,lon2=LonE)[0]# short-wave indident (W/m**2)
			Kil=grbs[275].data(lat1=LatN,lat2=LatN,lon1=LonE,lon2=LonE)[0] # longwave indient (W/m**2)]
			if 'd' in locals():
				d=np.vstack([d,np.array([date,lats[0][0],lons[0][0],T2m[0][0],H2m[0][0],Kis[0][0],Kil[0][0]])])#vstack (add new row) for each file
			else:
				d=np.array([date,lats[0][0],lons[0][0],T2m[0][0],H2m[0][0],Kis[0][0],Kil[0][0]])


dname=['date','lat','lon','T2m','H2m','Kis','Kil']
df=pandas.DataFrame(d)#transform in data frame
df.to_csv(OutPath+Outfilename+'.csv',header=dname,sep='\t')# write into a .csv format
print("=====")
print("outfile:" + OutPath+Outfilename+".csv")
print("sucessful!!!")

