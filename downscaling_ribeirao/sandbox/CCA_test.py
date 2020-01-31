#===============================================================================
# Create 2 dataset with the East and West stations and apply the CCA 
#===============================================================================
from statmod_lib import *

from sklearn.cross_decomposition import CCA


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
    West_stanames =AttSta.stations(['West'])
    East_stanames =AttSta.stations(['East'])
    
    
    AttSta.setInPaths(InPath)
    West_Files =AttSta.getatt(West_stanames,'InPath')
    East_Files =AttSta.getatt(East_stanames,'InPath')
    
    net_West = LCB_net()
    net_East = LCB_net()
    
    net_West.AddFilesSta(West_Files)
    net_East.AddFilesSta(East_Files)
    X = net_West.getvarallsta(var=var, by='H', how='mean', From=From, To=To)
    Y = net_East.getvarallsta(var=var, by='H', how='mean', From=From, To=To)

    cca = CCA(n_components=2)
    cca.fit(X, Y)

    X_c, Y_c = cca.transform(X, Y)
    print Y
    
    
    print "=="*20
    print "X"
    print "=="*20
    print X
    print X.shape

    print "=="*20
    print "X_c"
    print "=="*20
    print X_c
    print X_c.shape

    print "=="*20
    print "Coeff"
    print "=="*20
    print cca.coef_
    print cca.coef_.shape
    
    print "=="*20
    print "x Loadings"
    print "=="*20
    print cca.x_loadings_
    print cca.x_loadings_.shape
    
    print "=="*20
    print "x rotations"
    print "=="*20
    print cca.x_rotations_
    print cca.x_rotations_.shape

    print "=="*20
    print "x scores ==  x_c"
    print "=="*20
    print cca.x_scores_
    print cca.x_scores_.shape
    
    print "=="*20
    print "x weights"
    print "=="*20
    print cca.x_weights_
    print cca.x_weights_.shape
    
    print "Correlation coefficient"
    print np.corrcoef(X_c.T, Y_c.T)[0, 1]
    
    
    plt.plot(X_c)
    plt.plot(Y_c)
    plt.plot(cca.x_scores_)
    plt.show()

    plt.scatter(X_c, Y_c)
    plt.show()
    
    plt.plot(cca.x_loadings_)
    plt.show()
    
    
    plt.plot(cca.x_weights_)
    plt.show()
    
