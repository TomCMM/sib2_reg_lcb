#===============================================================================
# DESCRIPTION
#    Make a Spectral Analysis of the wind components at the station
#===============================================================================


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
import math

#===============================================================================
# Example taken from the scipy documentation 
#===============================================================================
# https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html

# # Number of sample points
# N = 600
# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# import matplotlib.pyplot as plt
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
# plt.show()


def PolarToCartesian(norm,theta, rot=None):
    """
    Transform polar to Cartesian where 0 = North, East =90 ....
    From where the wind is blowing !!!!!
    
    ARG:
        norm: wind speed (Sm m/s)
        theta: wind direction (Dm G)
        rot: Rotate the axis of an angle in degree
    RETURN:
        U,V
    """
    if not rot:
        U=norm*np.cos(map(math.radians,-theta+270))
        V=norm*np.sin(map(math.radians,-theta+270))
    else:
        U=norm*np.cos(map(math.radians,-theta+270+rot))
        V=norm*np.sin(map(math.radians,-theta+270+rot))
    return U,V


if __name__ == '__main__':
    df_sm = pd.read_csv("/home/thomas/phd/climaobs/data/sta_data/Sm.csv", index_col=0, parse_dates=True)
    y = df_sm['C07'].dropna()
 
##############################3
    # it could be replaced by the index of the dataframe df_sm.index
    N = len(y) # number of sampling points
    T = 2 # minutes
    x = np.linspace(0.0, N*T, N+1)
########################################
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.grid()
    plt.show()
    
