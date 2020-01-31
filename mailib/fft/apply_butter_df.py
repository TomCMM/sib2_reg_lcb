#===============================================================================
# Description
#    Return low (<D-1) and high (>D-1) frequency dataframe for each variables
# References
#     Butter filter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter
# Example
# https://dsp.stackexchange.com/questions/19084/applying-filter-in-scipy-signal-use-lfilter-or-filtfilt
#===============================================================================
from __future__ import division, print_function
import numpy as np
from numpy.random import randn
from numpy.fft import rfft
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft, ifft, fftfreq
import glob 
from os.path import basename

def plot_butter(sig, sigff, noise):
    plt.subplot(3, 1, 1)
    plt.title('')
    plt.plot(sig, color='silver', label='Original')
    plt.plot(sig_ff, color='#3465a4', label='Filtered')
    plt.legend(loc="best")
    
    plt.subplot(3, 1, 2)
    plt.plot(noise, color='purple', label='noise')
    plt.legend(loc="best")
   
    #==========================================================================
    # Fourier transform
    #===========================================================================
    yf = fft(sig)
    yf_noise = fft(noise)
    yf_filtred = fft(sig_ff)
    
    N = len(sig)
    T = 3600
    freq = fftfreq(N, T) # frequency in seconds
    
    idxs = np.arange(1, N // 2) # half of the index
    plt.subplot(3, 1, 3)
    x_plot = freq[idxs] # take the first sequence 
    y_plot = abs(yf[idxs] + abs(yf[-idxs]))/N
    y_plot_noise = abs(yf_noise[idxs] + abs(yf_noise[-idxs]))/N
    y_plot_filtred = abs(yf_filtred[idxs] + abs(yf_filtred[-idxs]))/N
    
    plt.semilogx(x_plot, y_plot, color='silver', label='Original')
    plt.semilogx(x_plot, y_plot_noise, color='purple', label='Noise')
    plt.semilogx(x_plot, y_plot_filtred, color='#3465a4', label='Filtered signal')
    plt.legend(loc="best")
    plt.xlabel('Frequence in Hertz')
    plt.show()



def filter(col,b,a):
    sigff = signal.filtfilt(b,a,col)
    return sigff




if __name__ == '__main__':
    #===========================================================================
    # Apply on observational data
    #===========================================================================
    inpath = '../../fill/out/' 
    outpath = '../out/'
    outpath_plot = '../res/butter/'
    
    files = glob.glob(inpath +"*")
    filter_order = 3
    Wn = 0.03 
    b, a = signal.butter(filter_order, Wn=0.045, analog=False) 

    for f in files:
        print(f)
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        # delselect =  df.filter(regex='SB').columns
        # print(delselect)
        # df = df.drop(delselect, 1)
#         print(df.loc[:, delselect])  
#         del df[delselect]
        df_ff = df.apply(filter, args=(b,a), axis=0)
        df_noise = df - df_ff

        df_ff.to_csv(outpath +'low/'+ basename(f)[0:2] + '.csv')
        df_noise.to_csv(outpath +'high/' +basename(f)[0:2] + '.csv')

        print(basename(f)[0:2])
        df_noise.plot()
        plt.savefig(outpath_plot + basename(f)[0:2] + '_high.png', legend=None)

        df_ff.plot()
        plt.savefig(outpath_plot+ basename(f)[0:2] + '_low.png', legend=None)
        
#         plt.show()
#         sig_ff = (b, a, sig)
#         noise = sig - sig_ff
#         plot_butter(sig, sig_ff, noise)




    
    
    
