#===============================================================================
# TEST ICA on the LCB data
#===============================================================================

from statmod_lib import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA


if __name__ == "__main__":
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
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
    X = net.getvarallsta(var='Pa H', by='H', how='mean', From=From, To=To)
    X.plot()
    plt.show()
#     # Compute ICA
#     ica = FastICA(n_components=3)
#     S_ = ica.fit_transform(X)  # Reconstruct signals
#     A_ = ica.mixing_  # Get estimated mixing matrix
# 
#     plt.figure()
#     plt.plot(S_)
# 
#     # For comparison, compute PCA
#     pca = PCA(n_components=3)
#     H = pca.fit_transform(X)
# 
#     plt.figure()
#     plt.plot(H)
#     plt.show()
