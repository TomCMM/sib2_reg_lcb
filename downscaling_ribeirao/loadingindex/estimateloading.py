#===============================================================================
# DESCRIPTION
#    THIS FILE TRY TO ESTIMATE THE LOADING OBTAIN FROM THE PCA APPLIED ON THE STATIONS OBSERVATION AT EACH ARPS MODEL GRIDPOINT
#
#
#
#    TODO :     I SHOULD PUT ALL THIS FUNCTION IN AN OBJECT WHICH AIM TO MANIPULATE DTAFRAME FROM THE MODEL


#===============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPRegressor
from clima_lib.LCBnet_lib import *
import pickle
from sklearn.feature_selection import RFE
from arps_lib.verification_df_lib import find_idxs_at_stations_positions, stations_in_arps_domain


if __name__=='__main__':

    #===========================================================================
    # READ  THE DATA
    #===========================================================================
#     loading = pd.read_csv("/home/thomas/phd/statmod/data/loadingindex/loadings_pca_sta_coldpool.csv", index_col=0)
#     loading_T = pd.read_csv("/home/thomas/phd/statmod/data/loadingindex/loadings_pca_sta_coldpool_T.csv", index_col=0)
#     scores = pd.read_csv("/home/thomas/phd/statmod/data/loadingindex/scores_pca_sta_coldpool.csv", index_col=0)
#     scores_T = pd.read_csv("/home/thomas/phd/statmod/data/loadingindex/scores_pca_sta_coldpool_T.csv", index_col=0)
#     
#     pc_component = pd.HDFStore('/home/thomas/phd/statmod/data/model_data/pc_component.h5')
#     loadings_model = pc_component['pc_component']
# 
#     loading_true = pd.read_csv("/home/thomas/phd/statmod/data/loadingindex/loadings_pca_sta_coldpool_fullperiod.csv", index_col=0)
#     scores_true = pd.read_csv("/home/thomas/phd/statmod/data/loadingindex/scores_pca_sta_coldpool_fullperiod.csv", index_col=0)

#     loading_true = pickle.load( open(  "/home/thomas/phd/statmod/data/loadingindex/loadings.p", "rb" ) ).T
#     loading_true.columns = [ str(c+1) for c in range(len(loading_true.columns))]


#     loading_true = loading.T
#     print loading_true
#===============================================================================
# Selection position stations
#===============================================================================

latgrid = pd.read_csv('/home/thomas/phd/statmod/data/model_data/sim_coldpool/latgrid.csv', index_col=0).T
longrid = pd.read_csv('/home/thomas/phd/statmod/data/model_data/sim_coldpool/longrid.csv', index_col=0).T
ZP = pd.read_csv('/home/thomas/phd/statmod/data/model_data/sim_coldpool/ZP.csv', index_col=0).T
latlon_model = pd.read_csv("/home/thomas/phd/statmod/data/model_data/sim_coldpool/latlon.csv", index_col=0, parse_dates=True)


#     stanames = loading_true.index
AttSta = att_sta()
stanames = AttSta.stations(['Ribeirao'])
print stanames
stalats = pd.Series(AttSta.getatt(stanames, "Lat"), name='Lat', index=stanames)
stalons = pd.Series(AttSta.getatt(stanames, "Lon"), name ='Lon', index=stanames)
stalatlon = pd.concat([stalats, stalons], axis=1)

print stalatlon

stalatlons = stations_in_arps_domain(stalatlon,latlon_model)
idx_sta_in_model = find_idxs_at_stations_positions(latgrid, longrid, stalatlons)
 
#===============================================================================
# TEST plot temperature
#===============================================================================
Tk = pd.HDFStore('/home/thomas/phd/statmod/data/model_data/sim_coldpool/Tk_store.h5')
 
Tk_at_stations = []
for key in Tk.keys():
    print key
    Tk_at_stations.append( Tk[key].loc[idx_sta_in_model,:].T)
     
Tk_at_stations = pd.concat(Tk_at_stations)
lat_at_station = latgrid.iloc[idx_sta_in_model.values,:]
lon_at_station = longrid.iloc[idx_sta_in_model.values,:]
ZP_at_station = ZP.iloc[idx_sta_in_model.values,:]
 
Tk_at_stations.columns = idx_sta_in_model.index
lat_at_station.index = idx_sta_in_model.index
lon_at_station.index = idx_sta_in_model.index
ZP_at_station.index = idx_sta_in_model.index
     
     
    print Tk_at_stations
    print lat_at_station
    print lon_at_station
    print ZP_at_station
 

 
     
Tk_at_stations.index = pd.to_datetime(Tk_at_stations.index)
Tk_at_stations.columns = idx_sta_in_model.index
Tk_at_stations = Tk_at_stations.sort_index()
Tk_at_stations.drop_duplicates(inplace=True)

import datetime
df_Tk = Tk_at_stations
df_Tk = df_Tk-273.15
df_Tk = df_Tk.loc['2015-07-29 11:45:00':'2015-07-31 18:00:00']
#     df_Tk = df_Tk.between_time('13:00','16:00')
# df_Tk = (df_Tk - df_Tk.mean()) / (df_Tk.max() - df_Tk.min())
df_Tk.index = df_Tk.index + pd.DateOffset(hours=15)
df_Tk = df_Tk.groupby(lambda x: x.hour).mean()
plt.figure()
plt= df_Tk.loc[:,['C10','C04','C05','C06','C07','C08','C09']].plot(color=['b','b','b','r','r','r','r'])
plt.xlabel('time')
plt.ylabel('Temperature (C)')
plt.show()
# 
# plt.show()
# #     
# #     
# # #     
# # #     
# # #     print df_Tk
# # #     
# # #     
# # # #     print df_norm
# # #     
# # # #     print Tk_at_stations
# # # #     print Tk_at_stations.index
# # # #     
# # # #     daily = Tk_at_stations.groupby(lambda t: (t.hour)).mean()
# # # #     
# # # #     daily.loc[:,['C10','C04','C05','C06','C07','C08','C09']].plot()
# # # #     plt.show()
# # # #         
# # #     #===========================================================================
# # #     # BASIC PLOT
# # #     #===========================================================================
# # #     print "Loading from the model"
# #       
# #     print "Loading from the station observation"
# #     print loading_true
# #     print df_Tk
# # #     
#     for PC in range(5):
#         plt.scatter(df_Tk, loading_true.loc[df_Tk.index ,PC+1])
#          
#         for i, txt in enumerate(df_Tk.index):
#             plt.annotate(txt, (df_Tk[txt], loading_true.loc[txt ,PC+1]))
#         plt.show()
# # #    
# # # #     #===============================================================================
# # # #     # PREPROCESSING
# # # #     #===============================================================================
# # # #    
# # #     X = loadings_model.iloc[idx_sta_in_model, 0:4]
# # #     print X
# # #     Y = loading_true.loc[idx_sta_in_model.index,'2']
# # #     # keep 20% for the test dataset
# # #     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
# # # #     X_train, X_test, Y_train, Y_test == X, X , Y, Y
# # # #       
# # # #    
# # # #     #===========================================================================
# # # #     # ESTIMATION THROUGH LINEAR REGRESSION
# # # #     #===========================================================================
# # #     reg = LinearRegression()
# # # #     reg.fit(X_train, Y_train)
# # #     reg.fit(X, Y)
# # # 
# # # 
# # # 
# # #     #===========================================================================
# # # #     # NEURAL NETWORK ESTIMATION
# # # #     #===========================================================================
# # # #     reg = MLPRegressor(hidden_layer_sizes=(10,2), solver="lbfgs", alpha=1e-5, random_state=1)
# # # #     reg.fit(X_train,Y_train)
# # # #  
# # #     #===============================================================================
# # #     # PLOT RESULTS
# # #     #===============================================================================
# # #    
# # #     plt.scatter(Y, reg.predict(X), c ='g', label='train dataset')  
# # #     for i, txt in enumerate(idx_sta_in_model.index):
# # #         plt.annotate(txt, (Y[i], reg.predict(X)[i]))
# # #     plt.xlabel('observed')
# # #     plt.ylabel('calculated')
# # #     plt.show()
# # # #  
# # #     print reg.score(X_test, Y_test)
# # #      
# # #     plt.scatter(Y_train, reg.predict(X_train), c ='g', label='train dataset')
# # #     plt.scatter(Y_test, reg.predict(X_test), c='r', label='test dataset')
# # #     plt.show()
# # # #     
# # #  
# # #          
# # # 
# # #     