#===============================================================================
# DESCRIPTION
#        Perform the bootstrapping of the climatic station from the different network
#===============================================================================


from bootstrap import FillGap
import glob
from LCBnet_lib import *



if __name__=='__main__':
    #===========================================================================
    # IAC 
    #===========================================================================
    InPath ='/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/obs_clean/IAC/'
    OutPath = '/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/full_2013_2016/'
    Files = glob.glob(InPath+"*")
     
    net = LCB_net()
    AttSta = att_sta()
    AttSta.setInPaths(InPath)
    stanames = AttSta.get_sta_id_in_metadata(['IAC'])
    distance = AttSta.dist_matrix(stanames)
  
    staPaths = AttSta.getatt(stanames , 'InPath')
    net.AddFilesSta(staPaths, net='IAC')
  
    From='2013-01-01 00:00:00' 
    To='2016-01-01 00:00:00'
     
    gap = FillGap(net)
    gap.fillstation([],all=True, variables=['Ta C'], From=From, To=To, by='H', how='mean',
                    summary=False, distance=distance,plot=False, constant=True, sort_cor=True)
#    
    gap.WriteDataFrames(OutPath)

    #===========================================================================
    # Innmet 
    #===========================================================================
#     InPath ='/home/thomas/PhD/obs-lcb/staClim/INMET-Master/obs_clean/INMET/'
#     OutPath = '/home/thomas/PhD/obs-lcb/staClim/INMET-Master/full/'
#     Files = glob.glob(InPath+"*")
#     
#     net = LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(InPath)
#     stanames = AttSta.stations(['Innmet'])
#     print stanames
#     distance = AttSta.dist_matrix(stanames)
#  
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     print staPaths
#     net.AddFilesSta(staPaths, net='INMET')
#  
#     From='2013-01-01 00:00:00' 
#     To='2016-01-01 00:00:00'
#     
#     gap = FillGap(net)
#     gap.fillstation([],all=True, variables=['Ta C'], From=From, To=To, by='H', how='mean',
#                     summary=False, distance=distance,plot=False, constant=True, sort_cor=True)
# #    
#     gap.WriteDataFrames(OutPath)