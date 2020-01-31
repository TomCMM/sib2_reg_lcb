from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    

#===========================================================================
# Observational data
#===========================================================================

inpath = '../../fill/out/' 
outpath = '../out/'
df = pd.read_csv(inpath + "Ta.csv", index_col=0, parse_dates=True)
input_signal = df.loc[:,'C07'].resample('H').mean()


#===============================================================================
# Example
#===============================================================================
fs = 1000.0  # Sample frequency (Hz)
f0 = 300.0  # Frequency to be retained (Hz)
Q = 30.0  # Quality factor

w0 = f0/(fs/2)  # Normalized Frequency

b, a = signal.iirpeak(w0, Q) # Design peak filter
w, h = signal.freqz(b, a)  # Frequency response
freq = w*fs/(2*np.pi) # generate frequency axis

plt.title('Digital filter frequency response')
plt.plot(w, 20*np.log10(np.abs(h)))
plt.title('Digital filter frequency response')
plt.ylabel('Amplitude Response [dB]')
plt.xlabel('Frequency (rad/sample)')
plt.grid()
plt.show()


#===============================================================================
# Butter application
#===============================================================================
b, a = signal.butter(N, Wn=0.5, 'low')
output_signal = signal.filtfilt(b, a, input_signal)





