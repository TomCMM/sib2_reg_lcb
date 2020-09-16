#===============================================================================
#    DECRIPTION
#     Export a dataframe to adas format
#===============================================================================
import pandas as pd
import numpy as  np 
import datetime

#===============================================================================
# import variables to be added and latitude
#===============================================================================
inpath = "/home/thomas/"
OutFilename='surface'
varname='PT'
sep = 7
date=datetime.datetime.strptime("2015-10-04", '%Y-%m-%d')# date of the file
hour = datetime.datetime.strptime("15:00", '%H:%M')# date of the file

lat  = np.loadtxt(inpath+"latitude.txt",delimiter=',')
lon  = np.loadtxt(inpath+"longitude.txt",delimiter=',')
var  = np.loadtxt(inpath+"variable.txt",delimiter=',')
elev  = np.loadtxt('/home/thomas/100_3rasterelev.txt',delimiter=',')


lat = pd.Series(lat.flatten())
lon = pd.Series(lon.flatten())
var = pd.Series(var.flatten())
elev = pd.Series(elev.flatten())


df = pd.concat([lat, lon, var, elev], axis=1)
df.columns = ['lat', 'lon', 'var', 'elev']

nobs = len(df)

#===============================================================================
# Header
#===============================================================================
#======= Write file 
f_out=open(inpath+OutFilename+'.lso', 'w')


#file header
f_out.write(" "+"{} {} {} {}{} {}\n".format(date.strftime('%d-%b-%Y'),hour.strftime('%H:%M:%S')+'.00',str(0).rjust(5),str(0).rjust(4),str(nobs).rjust(sep)*7,9999))


for i in df.index:
    #station header
    f_out.write("{} {} {} {} {} {} {}\n".format(str(varname).rjust(5),np.around(df['lat'][i],decimals=2),str(np.around(df['lon'][i],decimals=2)).rjust(sep),str(str(int(df['elev'][i]))+'.').ljust(5,'0'),str("SA").rjust(2),str(hour.strftime('%H%M')).rjust(10), "".rjust(8) ))
    #Data variable:line1
    f_out.write(" {} {} {} {} {} {} {} {} {}\n".format(str(np.round(df['var'][i],decimals=1)).rjust(9),str(-99.9).rjust(6),str(-99.9).rjust(5),str(-99.9).rjust(5),str(-99.9).rjust(5),str(-99.9).rjust(5),str(-99.9).rjust(6),str(-99.9).rjust(6),str(-99.9).rjust(6)))
    #Data variable:line2
    f_out.write("{} {} {} {} {} {} {}\n".format(str(0).rjust(6),str(-99.9).rjust(7),str(-99.9).rjust(7),str(-99.9).rjust(5),str("-99.900").rjust(7),str(-99.9).rjust(6),str(-99).rjust(4)))
   

f_out.close()

