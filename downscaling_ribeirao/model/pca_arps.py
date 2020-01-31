

import pandas as pd
from sklearn.decomposition import PCA
import pickle 
import matplotlib.pyplot as plt
import numpy as np


def Tk(P, PT):
    """
    Calculate the Real Temperature in Kelvin
    """
    print("Calculate the Real temperature in Kelvin")
    P0 = 100000
    Rd = 287.06
    Cp = 1004.5
    Tk = PT*(P/P0)**(Rd/Cp)
    return Tk


if __name__=='__main__':
    #===========================================================================
    # Scores PCA sta
    #===========================================================================
#     score_sta = pd.read_csv('/home/thomas/scores_pca_sta.csv', index_col=0, parse_dates=True )
#     print score_sta    
    
    
    def rdf(path):
        dfPT = pd.read_csv(path)
        dfPT =dfPT.T
        dfPT = dfPT.iloc[1:,:]
        dfPT.index = dfPT.index.to_datetime()
        return dfPT

    files = [
             "~/phd/statmod/data/model_data/P_arps_1.csv",
            "~/phd/statmod/data/model_data/P_arps_2.csv",
#              "~/phd/statmod/data/model_data/P_arps_3.csv",
#              "~/phd/statmod/data/model_data/P_arps_4.csv",
#              "~/phd/statmod/data/model_data/24fev/P_arps_1finaly.csv",
#              "~/phd/statmod/data/model_data/24fev/P_arps_2finaly.csv",
#              "~/phd/statmod/data/model_data/24fev/P_arps_3finaly.csv",
#              "/home/thomas/phd/statmod/data/model_data/2march/P_arps_1finaly.csv",
#              "/home/thomas/phd/statmod/data/model_data/2march/P_arps_2finaly.csv",
#              "/home/thomas/phd/statmod/data/model_data/2march/P_arps_3finaly.csv",
#              "/home/thomas/phd/statmod/data/model_data/2march/P_arps_somepartsindados3finaly.csv",
#              "/home/thomas/phd/statmod/data/model_data/2march/P_arps_lastpart.csv"
             ]

#     p1 = '/home/thomas/P_arps_1.csv'
#     p2 = '/home/thomas/P_arps_2.csv'
#     p3 = '/home/thomas/P_arps_3.csv'
#     p4 = '/home/thomas/P_arps_4.csv'
#     df = pd.DataFrame()
#     for file in files:
#         pd.concat([df,pd.DataFrame])
#     ps = [p1,p2,p3,p4]

#     df = pd.DataFrame()
#     for file in files:
#         print file
#         newdf = rdf(file)
#         df = df.append(newdf)
#         df.drop_duplicates(inplace=True)
#         print df.shape
# 
#     df.to_csv('~/phd/statmod/data/model_data/2march/P_full.csv')
#     print "Done"
        
#     dfPT1 = pd.read_csv('/home/thomas/PT_arps_1.csv')
#     for i in np.arange(1,4):
#     dfPT1 = pd.read_csv('/home/thomas/PT_arps_1.csv')
#     dfPT2 = pd.read_csv('/home/thomas/PT_arps_2.csv')
# #     dfPT3 = pd.read_csv('/home/thomas/PT_arps_3.csv')
# #     dfPT4 = pd.read_csv('/home/thomas/PT_arps_4.csv')
#     
#     dfPT = pd.concat([dfPT1,dfPT2],axis=0)
#     print df
    
#     
#     dfPT =dfPT.T
#     dfPT = dfPT.iloc[1:,:]
#     dfPT.index = dfPT.index.to_datetime()
#      
#     dfP = pd.read_csv('/home/thomas/P.csv')
#     dfP =dfP.T
#     dfP = dfP.iloc[1:,:]
#     dfP.index = dfP.index.to_datetime()
# 
# #     
# #     pt = f.variables['PT'][0,0,:,:]
# #     p = f.variables['P'][0,0,:,:]
#     df = Tk(dfP, dfPT)
#     df = df-273.15
#     print df
#      
#     df = df.groupby(lambda t: (t.hour)).mean()
#     score_sta = score_sta.groupby(lambda t: (t.hour)).mean()
# 
#     c = df.corrwith(score_sta.iloc[:,2])
# #     c.to_csv('/home/thomas/arps_corr_PC2.csv')
#  
# #      
#     mapcorr = np.reshape(c, (1201,1201))
#     levels = np.linspace(mapcorr.min(), mapcorr.max(), 100 )
#     plt.contourf(mapcorr, levels = levels, cmaps = 'plasma')
#     plt.colorbar()
#     plt.show()
# # #     
#     df_merge = pd.concat([df, score_sta], join='inner', axis=1)
#     print df_merge.corr()
    
#     from pandas.tools.plotting import scatter_matrix
#     scatter_matrix(df_merge, alpha=0.2, figsize=(6, 6), diagonal='kde')
    
#     print "Correlation matrix"
#     c = df.iloc[:,1:10].corrwith(score_sta, axis=1)
#     print c
# #     df = df.iloc[:,::1000]
# #     del df[0,:]
#      
#  
# # #     pickle.dump( df.T, open( "/home/thomas/PT.p", "wb" ) )
# #     df = pickle.load( open( "/home/thomas/PT.p", "rb" ) )
# #     print df
#         
#     #===========================================================================
#     # TEST PCA
#     #===========================================================================
#           
#     nbpc = 6
# #     nbpts = 200
#     pca = PCA(n_components=nbpc)
#          
#        
# #     df = (df - df.mean(axis=0)) / df.std(axis=0)
# #     print df
#     
#     df_scores = pca.fit_transform(df)
#  
#     
#     
#     plt.plot(df_scores)
#     plt.show()
#         
#     loadings = pca.components_
#       
#     print loadings.shape
#       
#       
#     for pc in range(nbpc):
#         pcload = np.reshape(loadings[pc,:], (1201,1201))
#           
#         levels = np.linspace(pcload.min(), pcload.max(), 100)
#         plt.contourf(pcload, levels = levels)
#         plt.colorbar()
#         plt.show()
# #  
#       
      
# 
#     for pc in range(nbpc -1):
#         plt.scatter(loadings[pc,:],loadingsmap[pc,:55], alpha=0.10)
#         plt.xlabel("PC sta "+str(pc+1))
#         plt.ylabel("PC sta taken with map"+str(pc+1))
#         plt.show()
#     