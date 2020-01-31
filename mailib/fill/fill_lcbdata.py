#==============================================================================
#   Description
#       module which fill the missing data of the station observations through multiregression analysis
#    Update
#        August 2016:
#            
#==============================================================================


# Library
from __future__ import division
from stadata.lib.LCBnet_lib import *
from mailib.fill.fill import FillGap

if __name__=='__main__':

#===============================================================================
# Bootstrap data - article
#===============================================================================
#     InPath ='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
#     OutPath = '/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     Files = glob.glob(InPath+"*")
#     
#     net = LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(InPath)
#     stanames = AttSta.stations(['Ribeirao'])
#     distance = AttSta.dist_matrix(stanames)
# 
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
# 
#     From='2014-10-15 00:00:00' 
#     To='2016-08-01 00:00:00'
#    
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='mean',
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#   
#     gap.WriteDataFrames(OutPath)


#===============================================================================
# Bootstrap data - article
#===============================================================================
    InPath ='/home/thomas/phd/obs/lcbdata/obs/Merge/'
    OutPath = '/home/thomas/phd/obs/lcbdata/obs/full_2min/'
    Files = glob.glob(InPath+"*")
    
    net = LCB_net()
    AttSta = att_sta()
    AttSta.setInPaths(InPath)
    stanames = AttSta.stations(['Ribeirao'])
    print(stanames)
    distance = AttSta.dist_matrix(stanames)

    staPaths = AttSta.getatt(stanames , 'InPath')
    net.AddFilesSta(staPaths)

    From='2015-01-01 00:00:00'
    To='2016-06-01 00:00:00'
   
    gap = FillGap(net)
    gap.fillstation([], all = True, From=From, To=To, by='H', how='mean',
                    summary=False, plot=False, distance=distance, constant=True, sort_cor=True)
  
    gap.WriteDataFrames(OutPath)



#==============================================================================
# Specific bootstraping for Rainfall - TEST
#==============================================================================
#     InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
#     OutPath= '/home/thomas/PhD/obs-lcb/LCBData/obs/Full_nearest/'
#     OutPath2= '/home/thomas/PhD/obs-lcb/LCBData/obs/Full_nearest2/'
#     OutPath3= '/home/thomas/PhD/obs-lcb/LCBData/obs/Full_nearest3/'
# #     OutPath3= '/home/thomas/PhD/obs-lcb/LCBData/obs/Full_Allperiod_medio/'
# #     Files=glob.glob(InPath+"*")
#      
# #     stanames = stanames# +['C12', 'C04']
# #     stanames = ['C10','C11', 'C12']
#      
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(InPath)
#     stanames = AttSta.stations(['Ribeirao'])
# #     stanames = ['C04','C05','C06','C07','C13','C14','C15']
# 
# #     [stanames.remove(x) for x in ['C05','C13'] if x in stanames ]
# #     stanames = stanames +['C12','C10','C05' 'C04']
#      
#      
#     distance = AttSta.dist_matrix(stanames)
#     print distance
#     
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#        
# #     # Middle
#     From='2014-10-15 00:00:00' 
#     To='2016-04-01 00:00:00'
#   
#     #Head
# #     From='2015-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#            
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#  
#     gap.WriteDataFrames(OutPath)
# #     
# # #------------------------------------------------------------------------------ 
# # #     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#       
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#        
# #      
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#                
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#      
#     gap.WriteDataFrames(OutPath2)
# # 
# # #------------------------------------------------------------------------------ 
# #     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2)
#    
#    
# #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2)
#    
# #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#    
# #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#     
#     #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#     
#     #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#     
#     #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#     
#     #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#     
#     #------------------------------------------------------------------------------ 
#     # Perform another bootstrapping using the previous bootstraped data as input
#     net=LCB_net()
#     AttSta = att_sta()
#     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
#     distance = AttSta.dist_matrix(stanames)
#     staPaths = AttSta.getatt(stanames , 'InPath')
#     net.AddFilesSta(staPaths)
#      
#     # Middle
# #     From='2015-03-15 00:00:00' 
# #     To='2016-01-01 00:00:00'
#       
#         
# #     From='2014-11-01 00:00:00' 
# #     To='2016-01-01 00:00:00'
#               
#     gap = FillGap(net)
#     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
#                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False)
#     
#     gap.WriteDataFrames(OutPath2) 
#     
# # #  
# #------------------------------------------------------------------------------ 
# #     # Perform another bootstrapping using the previous bootstraped data as input
# #     net=LCB_net()
# #     AttSta = att_sta()
# #     AttSta.setInPaths(OutPath2)
# #     stanames = AttSta.stations(['Head'])
# #     distance = AttSta.dist_matrix(stanames)
# #     staPaths = AttSta.getatt(stanames , 'InPath')
# #     net.AddFilesSta(staPaths)
# #    
# #     # Middle
# # #     From='2015-03-15 00:00:00' 
# # #     To='2016-01-01 00:00:00'
# #     
# #       
# # #     From='2014-11-01 00:00:00' 
# # #     To='2016-01-01 00:00:00'
# #             
# #     gap = FillGap(net)
# #     gap.fillstation([], all = True, From=From, To=To, by='H', how='sum',variables = ['Rc mm'], 
# #                     summary=False, plot=False, distance=distance, constant=True, sort_cor=False, cor_lim=0.90)
# #   
# #     gap.WriteDataFrames(OutPath2) 



#===============================================================================
# Clean Full
#===============================================================================
 
#     InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#     OutPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
#          
#          
#     Files=glob.glob(InPath+"*")
#          
#     #     threshold={
#     #                 'Pa H':{'Min':850,'Max':920},
#     #                 'Ta C':{'Min':5,'Max':40,'gradient_2min':4},
#     #                 'Ua %':{'Min':0.0001,'Max':100,'gradient_2min':15},
#     #                 'Rc mm':{'Min':0,'Max':8},
#     #                 'Sm m/s':{'Min':0,'Max':30},
#     #                 'Dm G':{'Min':0,'Max':360},
#     #                 'Bat mV':{'Min':0.0001,'Max':10000},
#     #                 'Vs V':{'Min':9,'Max':9.40}}
#          
#     for f in Files:
#         print f
#         if f == "/home/thomas/PhD/obs-lcb/LCBData/obs/Full_new/C15.TXT":
#             df = pd.read_csv(f, sep=',', index_col=0,parse_dates=True)
#             print df
#             df['Ta C'][(df['Ta C']<5) | (df['Ta C']>35) ] = np.nan
#             df['Ta C'] = df['Ta C'].fillna(method='pad')
#             df['Ua %'][(df['Ua %']<=0) | (df['Ua %']>=100) ] = np.nan
#             df['Ua %'] = df['Ua %'].fillna(method='pad')
#             df['Ua g/kg'][(df['Ua g/kg']<0) | (df['Ua g/kg']>25) ] = np.nan
#             df['Ua g/kg'] = df['Ua g/kg'].fillna(method='pad')
#             df.to_csv("/home/thomas/PhD/obs-lcb/LCBData/obs/Full_new_2/C15.TXT")
#         if f == "/home/thomas/PhD/obs-lcb/LCBData/obs/Full_new/C10.TXT":
#             print "allo"
#             df = pd.read_csv(f, sep=',', index_col=0, parse_dates=True)
#             df['Ta C'][(df['Ta C']<0) | (df['Ta C']>36) ] = np.nan
#             df['Ua %'][(df['Ua %']<0) | (df['Ua %']>100) ] = np.nan
#             df['Ua g/kg'][(df['Ua g/kg']<0) | (df['Ua g/kg']>25) ] = np.nan
#             df['Ua g/kg'] = df['Ua g/kg'].fillna(method='pad')
#             df['Ua %'] = df['Ua %'].fillna(method='pad')
#             df.to_csv("/home/thomas/PhD/obs-lcb/LCBData/obs/Full_new_2/C10.TXT")


    #===============================================================================
    # Interpolate Pressure C08
    #===============================================================================
    
#     from sklearn import datasets, linear_model
#     import statsmodels.api as sm
#     
#     
#     df9 = pd.read_csv("/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C09.TXT", sep=',', index_col=0,parse_dates=True)
#     df7 = pd.read_csv("/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C07.TXT", sep=',', index_col=0,parse_dates=True)
#     df8 = pd.read_csv("/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C08.TXT", sep=',', index_col=0,parse_dates=True)
#     
#     dfpa9 = df9['Pa H']
#     dfpa8 = df8['Pa H']
#     dfpa7 = df7['Pa H']
#     
#     dfPa = pd.concat([dfpa9,dfpa7,dfpa8], axis=1, join ='inner')
#     dfPa.columns = ['C09', 'C07', 'C08']
#     
#     # dfPa = dfPa[-50:]
#     # print dfPa
#     
#     elev9 = 1356
#     elev7 = 1186
#     elev8 = np.array([1225])
#          
#     def inter(row):
#         y = np.array([[row['C09']], [row['C07']]])
#         x= np.array([[elev9], [elev7]])
#         x = sm.add_constant(x)
#         regr = sm.OLS(y,x)
#         results = regr.fit()
#         return results.params[1]*elev8 + results.params[0]
#              
#     dfPa['C08_new'] = dfPa.apply(inter, axis=1)['C08']
#     df8 = pd.concat([df8, dfPa['C08_new']], axis=1)
#     df8['Pa H']=df8['C08_new']
#     df8 = df8.drop(['C08_new'],1)
#     df8.to_csv("/home/thomas/PhD/obs-lcb/LCBData/obs/Full/C08.TXT")
#          
#     df8[['Pa H', 'C08_new']].plot()
#     plt.show()




