#===============================================================================
# DESCRIPTION
#    Clean the files of every networks
#===============================================================================


import glob
from stadata.lib.LCBnet_lib import LCB_net
import pandas as pd
import os 
import fnmatch


if __name__ =='__main__':

#===============================================================================
# Sinda Merge
#===============================================================================
    
#     Path='/home/thomas/PhD/obs-lcb/staClim/Sinda/obs/'
#     OutPath ='/home/thomas/PhD/obs-lcb/staClim/Sinda/Sinda_merged/'
#        
#     Files = []
#     for root, dirnames, filenames in os.walk(Path):
#         for filename in fnmatch.filter(filenames, '*.csv'):
#             Files.append(os.path.join(root, filename))
#             # rename intor csv
#        
#     meta = pd.read_csv('/home/thomas/PhD/obs-lcb/staClim/database_metadata.csv', index_col=0)
#     meta_Sinda = meta[meta['network'] == "Sinda"]
#    
#     for i, sta in enumerate(meta_Sinda.index):
#    
#         df_merged = pd.DataFrame()
#         select = [s for s in Files if str(sta) in s]
#         print sta
#         for f in select:
#             if f[-9:-4] == sta:
#                 station = LCB_station(f, net='Sinda', clean=False )
#                 df_merged = pd.concat([df_merged, station.Data])
#    
#         if not df_merged.empty:
#             df_merged = df_merged.groupby(df_merged.index).first()
#             df_merged.to_csv(OutPath+sta+".csv")

#===============================================================================
# Sinda Clean
#===============================================================================
# 
#     metadata = pd.read_csv('/home/thomas/PhD/obs-lcb/staClim/database_metadata.csv', index_col=0)
#     Path='/home/thomas/PhD/obs-lcb/staClim/Sinda/Sinda_merged_cleanhand/'
#     Files=glob.glob(Path+"*")
#      
#     print Files
#     net=LCB_net()
#     net.AddFilesSta(Files, net='Sinda')
#      
#     #     data = net.getData('Ta C')
#     #     data.plot()
#     #     plt.show()
#     #     data = net.getvarallsta('Rc mm')
#     #     data.plot()
#     #     plt.show()
#     net.write_clean('/home/thomas/Sinda/')

#===============================================================================
# INNMET CLEAN
#===============================================================================
#     metadata = pd.read_csv('/home/thomas/PhD/obs-lcb/staClim/database_metadata.csv', index_col=0)
#     Path='/home/thomas/PhD/obs-lcb/staClim/INMET-Master/obs/'
#     Files=glob.glob(Path+"*")
# #         
# #     sta = LCB_station(Files[0],net='INMET' )
# #     sta.showpara()     
# #     print sta.Data
#    
#     print Files
#     net=LCB_net()
#     net.AddFilesSta(Files, net='INMET', clean=False)
#     
# #     data = net.getData('Ta C')
# #     data.plot()
# #     plt.show()
# #     data = net.getvarallsta('Rc mm')
# #     data.plot()
# #     plt.show()
#     net.write_clean('/home/thomas/PhD/obs-lcb/staClim/INMET-Master/obs_clean/')

#===============================================================================
# IAC MERGE
#===============================================================================
# 
#     meta = pd.read_csv('/home/thomas/PhD/obs-lcb/staClim/database_metadata.csv', index_col=0)
#     meta_IAC = meta[meta['network'] == "IAC"]
#     meta_IAC['sigle'] = meta_IAC.index.str[0:2]
#     print  meta_IAC
# 
#                 
#     # Create a list of all possible files
#     InPath='/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/obs/hourly/'
#     OutPath = '/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/obs/hourly_merged/'
#     Files = []
#     for s, s_complete in zip(meta_IAC['sigle'],meta_IAC.index) :
#         print s
# #     for s, s_complete in zip(['ex'],['ex23']) :
#         print "0"*80
#         print s
#         print "0"*80
#         df_merged = pd.DataFrame()
#         for y in [13,14,15,16]:
#             print y
#             try:
#                 path = InPath+str(y)+str(s)+"333.dat"
#                 sta = LCB_station(path,net='IAC', clean=False )
#                 df_merged = pd.concat([df_merged, sta.Data])
#             except IOError:
#                 print "File Does not exist"
#             except ValueError:
#                 print "Wrong number of columns"
#                    
#                             
#         if not df_merged.empty:
#             df_merged = df_merged.groupby(df_merged.index).first()
# #             df_merged = df_merged.drop_duplicates(subset='index', take_last=True)
#             df_merged.to_csv(OutPath+str(s_complete)+".csv")


#===============================================================================
# IAC Clean
#===============================================================================
#     Path= "/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/obs_clean/IAC/"
#     Files=glob.glob(Path+"*")
# #     Files = ['/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/obs/test_merged/ex23.csv']
#     
# #     for f in Files:
# #         print f
# #         sta = LCB_station(f,net='IAC' )
# #         sta.showpara()     
# #         print sta.Data
#  
#  
#      
#     print Files
#     net=LCB_net()
#     net.AddFilesSta(Files, net='IAC')
#         
#     net.write_clean('/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/obs_clean/IAC2/')

# #===============================================================================
# # metar
# #===============================================================================
#     Path= "/home/thomas/phd/obs/staClim/metar/data/data/"
#     Files=glob.glob(Path+"*")
# #     Files = ['/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/obs/test_merged/ex23.csv']
#       
# #     for f in Files:
# #         print f
# #         sta = LCB_station(f,net='IAC' )
# #         sta.showpara()     
# #         print sta.Data
#    
#    
#        
#     print Files
#     net=LCB_net()
#     net.AddFilesSta(Files, net='metar')
#           
#     net.write_clean("/home/thomas/phd/obs/staClim/metar/data/clean/")


# # #===============================================================================
# # # cetesb clean
# # #===============================================================================
    Path= "/home/thomas/phd/obs/staClim/metar/data/clean_select_byhand/"
    Files=glob.glob(Path+"*")
#     Files = ['/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/obs/test_merged/ex23.csv']
       
#     for f in Files:
#         print f
#         sta = LCB_station(f,net='IAC' )
#         sta.showpara()     
#         print sta.Data
    
    
        
    print Files
    net=LCB_net()
    net.AddFilesSta(Files, net='metar', clean=False)
           
    net.write_clean("/home/thomas/metar/")


#===============================================================================
# pcd clean
#===============================================================================
#     Path= "/home/thomas/phd/obs/staClim/pcd_serradomar/data/merge/"
#     Files=glob.glob(Path+"*")
# #     Files = ['/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/obs/test_merged/ex23.csv']
#      
# #     for f in Files:
# #         print f
# #         sta = LCB_station(f,net='IAC' )
# #         sta.showpara()     
# #         print sta.Data
#   
#   
#       
#     print Files
#     net=LCB_net()
#     net.AddFilesSta(Files, net='pcd', clean=False)
#          
#     net.write_clean("/home/thomas/phd/obs/staClim/pcd_serradomar/data/clean/")


