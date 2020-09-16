#===============================================================================
# DESCIRPTION
#    Plot the data from INNMET. IAC and SINDA with the purpose
#    to verify the consistency of the data
#===============================================================================

from stadata.lib.LCBnet_lib import LCB_net
import glob
import pandas as pd
from stadata.lib.time_serie_plot import TimeSeriePlot, att_sta
import matplotlib.pyplot as plt

if __name__ =='__main__':
    From = "2010-01-01 00:00:00"
    To = "2017-01-01 00:00:00"
     
    AttSta = att_sta()

#===============================================================================
# Daily average overall network IAC
#===============================================================================
#     Path_IAC ='/home/thomas/phd/obs/staClim/iac/data/IAC/'
# #     Path_IAC ='/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/obs_clean/IAC/'
# #     AttSta_IAC = att_sta(Path_att="/home/thomas/phd/obs/staClim/metadata/database_metadata.csv")
# #     AttSta_IAC.setInPaths(Path_IAC)
#     Files_IAC=glob.glob(Path_IAC+"*")
#     net=LCB_net()
# #     
#      
# #     stanames_IAC =  AttSta.stations(values=['IAC'])
# #     print stanames_IAC
# #     Files_IAC =AttSta_IAC.getatt(stanames_IAC,'InPath')
#     net.AddFilesSta(Files_IAC, net='IAC')
# #     print Files_IAC
#     TimeSeriePlot(Files_IAC,var=[ 'Dm G'], OutPath = '/home/thomas/IAC/DMG/')

#===============================================================================
# Time serie plot IAC
#===============================================================================
# 
#     Path='/home/thomas/phd/obs/staClim/iac/obs_clean/IAC/'
# #     Path='/home/thomas/MergeDataThreeshold/'
#     OutPath='/home/thomas/phd/obs/staClim/iac/fig/'
#     Files=glob.glob(Path+"*")
#     TimeSeriePlot(Files,var=[ 'Ua %'], OutPath = OutPath)

#===============================================================================
# Time serie plot INNMET
#===============================================================================


# 
#     Path ='/home/thomas/phd/obs/staClim/inmet/obs_clean/INMET/'
#     AttSta = att_sta(Path_att="/home/thomas/phd/obs/staClim/metadata/metadata_allnet_select.csv")
#     AttSta.setInPaths(Path)
#     Files=glob.glob(Path+"*")
#     net=LCB_net()
#  
#     stanames =  AttSta.stations(values=['Innmet'])
# #     print stanames
#     Files =AttSta.getatt(stanames,'InPath')
# #     print Files
#     net.AddFilesSta(Files, net='INMET')
# #     net.getvarallsta("Sm m/s").plot()
# #     plt.show()
# #      
#     TimeSeriePlot(Files,var=['Dm G'], OutPath ='/home/thomas/inmet/dmg/')
    
#===============================================================================
# Time serie plot Sinda
#===============================================================================
 
#     Path = '/home/thomas/PhD/obs-lcb/staClim/sinda/Sinda_merged_cleanhand/'
# #     Path='/home/thomas/MergeDataThreeshold/'
#     OutPath='/home/thomas/PhD/obs-lcb/staClim/sinda/fig/'
#     Files=glob.glob(Path+"*")
#     TimeSeriePlot(Files,var=[ 'Ta C'], OutPath = OutPath, net='Sinda')

# #===============================================================================
# # Time serie plot pcd_serra_do_mar
# #===============================================================================
# #  
#     Path = '/home/thomas/phd/obs/staClim/pcd_serradomar/data/clean/'
# #     Path='/home/thomas/MergeDataThreeshold/'
#     OutPath='/home/thomas/phd/obs/staClim/pcd_serradomar/fig/'
#     Files=glob.glob(Path+"*")
#     TimeSeriePlot(Files,var=[ 'Pa H'], OutPath = OutPath, net='Sinda')



# #===============================================================================
# # Time serie plot cetesb
# #===============================================================================
#  
#     Path = '/home/thomas/phd/obs/staClim/cetesb/data/clean/'
# #     Path='/home/thomas/MergeDataThreeshold/'
#     OutPath='/home/thomas/phd/obs/staClim/cetesb/fig/'
#     Files=glob.glob(Path+"*")
#     TimeSeriePlot(Files,var=[ 'Pa H'], OutPath = OutPath, net='cetesb')


#===============================================================================
# Time serie plot metar
#===============================================================================
#   
#     Path = '/home/thomas/phd/obs/staClim/metar/data/clean_select_byhand/'
# #     Path='/home/thomas/MergeDataThreeshold/'
#     OutPath='/home/thomas/phd/obs/staClim/metar/fig/'
#     Files=glob.glob(Path+"*")
#     try:
#         TimeSeriePlot(Files,var=['Pa H'], OutPath = OutPath, net='metar')
#     except ValueError:
#         pass

