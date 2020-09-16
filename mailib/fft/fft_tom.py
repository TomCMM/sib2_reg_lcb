#===============================================================================
# DESCRIPTION
#    Example of a low and high pass filter
#
# REFERENCES
#    https://stackoverflow.com/questions/19122157/fft-bandpass-filter-in-python
#===============================================================================


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fftpack import fft, ifft, fftfreq  # rfft and irfft are for real

from fft.fft_lib import  gettimestep


def check_time(df_index):
    """
    Check if the index is correctly set up
    """
    date  = pd.date_range(df_index[0], df_index[-1],  freq='1H')
    print date
    print df.index
    mask =  date
    if not any(mask):
        raise "Problem in the datettime index"
    else:
        return date
    
    
    return 

if __name__ == '__main__':
    
    inpath = '../../fill/out/' 
    outpath = '../out/'
    #===========================================================================
    # Read data
    #===========================================================================
    df = pd.read_csv(inpath + "Ta.csv", index_col=0, parse_dates=True)
    
    df = pd.read_csv("/home/thomas/phd/obs/lcbdata/obs/full/"+"C07.TXT", 
                     index_col=0, parse_dates=True)
#     df.index = check_time(df.index)
    
    y = df.loc['Ta C']
    
    y =(y-y.mean())/y.std()
    
    plt.plot(signal.detrend(y.values))
    plt.plot(y.values)
    plt.show()
#     # fft hello world
#     y = fft(df) # transform
#     y_inv = ifft(y) # invert
#     
    # Number of sample points
    N = len(y.index)
     
    # sample spacing 
    T =  gettimestep(y.index) # in seconds
     
    # Apply fourier transform
    yf = fft(y.values)
     
    freq = fftfreq(N, T) # frequency in seconds
    time_freq = 'seconds'
    
    
    #===========================================================================
    # Plot with scipy
    #===========================================================================
    idxs = np.arange(1, N // 2) # half of the index
    name = 'Temperature'
 
    plt.title("Fast fourier transform %s" % name)
    plt.xlabel("%s" % time_freq)
    plt.ylabel(" Amplitude m/s")
 
    x_plot = freq[idxs] # take the first sequence 
    y_plot = abs(yf[idxs] + abs(yf[-idxs]))/N
     
    plt.plot(x_plot, y_plot)
    plt.show()
   
    #===========================================================================
    # Low and high pass filter
    #===========================================================================
    low_freq = 0.5 # in days 
    high_freq = 0.1 # in days
    
    yf_low = yf.copy()
    yf_high = yf.copy()
    
    yf_low[(freq < low_freq)] = 0 
    yf_high[(freq > high_freq)] = 0
    
    yf_low_inv = ifft(yf_low) 
    yf_high_inv = ifft(yf_high)

    #===========================================================================
    # Plot
    #===========================================================================
    
    import pylab as plt
    plt.figure(figsize=(11.69,8.27))
    plt.subplot(321)
    plt.title('Original data')
    plt.plot(y.index, y)
    
    
    # we do not take the first value in the frequency domain as it much higher than the others
    plt.subplot(322)
    plt.title('Original data in the frequency domain')
    plt.plot(freq[1:], abs(yf[1:]))
    plt.xlim(0,10)
    plt.xlabel("%s" % time_freq)
    plt.ylabel(" Amplitude")

    plt.subplot(323)
    plt.title('Low pass filter data recontructed time serie')
    plt.plot(y.index, yf_low_inv)
 
    plt.subplot(324)
    plt.title('Low pass filter data in the frequency domain') 
    plt.plot(freq[1:],abs(yf_low[1:]))
    plt.xlim(0,10)
    plt.xlabel("%s" % time_freq)
    plt.ylabel(" Amplitude")

    plt.subplot(325)
    plt.title('High pass filter data recontructed time serie')
    plt.plot(y.index, yf_high_inv)

    plt.subplot(326)
    plt.title('High pass filter data in the frequency domain') 
    plt.plot(freq[1:],abs(yf_high[1:]))
    plt.xlim(0,10)
    plt.xlabel("%s" % time_freq)
    plt.ylabel(" Amplitude")

    plt.show() 
