#===============================================================================
# Description
#    Apply temporal filter to the framework dataframe
#===============================================================================


import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft, ifft, fftfreq  # rfft and irfft are for real

from fft.fft_lib import  gettimestep


def check_time(df_index):
    """
    Check if the index is correctly set up
    """
    date  = pd.date_range(df_index[0], df_index[-1],  freq='1H')
    print date
    mask =  df.index == date
    if not any(mask):
        raise "Problem in the datettime index"
    else:
        return date
    
    
    return 


def fft_df(y, freq_cut=0.9, low=True):
    """
    y: Hourly time serie 
    """
    print freq_cut
    print low

    # Number of sample points
    N = len(y.index)
     
    # sample spacing 
    T =  gettimestep(y.index) # in hours
        
    
    # Apply fourier transform
    yf = fft(y.values)
     
    freq = fftfreq(N, T/24.) # frequency in days
    time_freq = 'days'    
    
    yf_cut = yf.copy()
    
    if low:
        print "Low pass filter"
        yf_cut[(freq < freq_cut)] = 0
    else: 
        print "High pass fitler"
        yf_cut[(freq > freq_cut)] = 0
    
    yf_cut_inv = ifft(yf_cut) 
    
    return yf_cut_inv

if __name__ == '__main__':
    inpath = '../../fill/out/' 
    outpath = '../out/'
    #===========================================================================
    # Read data
#===========================================================================
    df = pd.read_csv(inpath + "Ta.csv", index_col=0, parse_dates=True)
    df.index = check_time(df.index)
    
#     df =(df-df.mean())/df.std()
    y = df.loc[:,'C07']
    res = fft_df(y, freq_cut=0.5, low=False)
    plt.plot(res)
    plt.show()
    
    res = fft_df(y, freq_cut=0.5, low=True)
    plt.plot(res)
    plt.show()
# freq_cut=0.5
# low=True
# df_low = df.apply(fft_df, axis=0, args=(freq_cut , low))
# 
# low=False
# df_high = df.apply(fft_df, axis=0, args=(freq_cut, low))
# 
# df_low.plot()
# plt.show()
# df_high.plot()
# plt.show()

    
    
    