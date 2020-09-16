from statmod_lib import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import KernelPCA, PCA, IncrementalPCA
import pandas as pd
import time
if __name__ == "__main__":
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full_sortcorr/'
    var = "Ta C"
    AttSta = att_sta('/home/thomas/PhD/obs-lcb/staClim/metadata_allnet.csv')
    From = "2015-03-01 00:00:00"
    To = "2016-01-01 00:00:00"
    Lat = [-25,-21]
    Lon = [-48, -45]
    Alt = [400,5000]

    print AttSta.showatt()
    stanames =AttSta.stations(['West'])

    print stanames
    AttSta.setInPaths(InPath)
    Files =AttSta.getatt(stanames,'InPath')

    net = LCB_net()
    net.AddFilesSta(Files)
    X = net.getvarallsta(var=var, by='H', how='mean', From=From, To=To)

#     X = X[0:100]
#     for i in range(17):
#         X = pd.concat([X,X], axis=1)
#     print X.shape

    tic = time.clock()
    pca = IncrementalPCA(n_components=2, batch_size=3)
    K = pca.fit_transform(X)
    toc = time.clock()
    print toc - tic

    #For comparison, compute PCA
    tic = time.clock()
    pca = PCA(n_components=2)
    H = pca.fit_transform(X)
    toc = time.clock()
    print toc - tic
    
    plt.figure()
    plt.plot(K)
    
    plt.figure()
    plt.plot(H)
    plt.show()
#     
#     print "done"
#     kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
#     X_kpca = kpca.fit_transform(X)
#     X_back = kpca.inverse_transform(X_kpca)
#     
#     plt.figure()
#     plt.plot(X_kpca)
#     plt.show()
#     
#     import numpy as np
# #     X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
#     from sklearn.decomposition import NMF
#     model = NMF(n_components=3, init='random', random_state=0)
#     A = model.fit_transform(X)
#     plt.figure()
#     plt.plot(A)
#     plt.show()
