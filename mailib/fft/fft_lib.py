
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, ifft


def gettimestep(index):
    """  input : Dataframe time index

        convert string time index into second

        output : integer """

    timestep = index[1] - index[0]
    timestep.seconds
    return timestep.seconds

def plotfft(serie, timestep, name, freq='d'):
    """ input: serie of data, timestep,

        argument freq : d for day, m for month, y for year, default in hour

        dependence : matplotlib.pyplot

        output: plot the serie with the frequency chosen"""


    n = len(serie)

    timestep_hour = float(timestep / 3600.)

    if freq == "d":

        freq = fftfreq(n, timestep_hour/24)
        time_freq = "Day -1"

    elif freq == "m":

        freq = fftfreq(n, timestep_hour/(24*31))
        time_freq = "Month -1"

    elif freq == "y":

        freq = fftfreq(n, timestep_hour / (24 * 365))
        time_freq = "Year -1"

    else:

        freq = fftfreq(n, timestep_hour)
        time_freq = "Hour -1"

    idxs = np.arange(1, n // 2)

    plt.title("Fast fourier transform %s" % name)
    plt.xlabel("%s" % time_freq)
    plt.ylabel(" Amplitude m/s")

    x = freq[idxs] # take the first sequence 
    y = abs(serie[idxs] + abs(serie[-idxs]))/n
    
    plt.plot(x, y)
    plt.show()


def low_filter(seriee, timestep, r_frq=0.9):
    """input: serie of data, timestep of the serie

        remove all freq below r_frq

        argument r_frq : the max frequency removed r_freq in Day -1

        dependence : scipy.fftpack fftfreq

        output: filtered np.array """

    freq = fftfreq(len(seriee), timestep/(3600*24))
    print freq.max()
    print freq.min()
    seriee[freq < r_frq] = 0

    return seriee


def high_filter(seriee, timestep, r_frq=1.1):
    """input: serie of data, timestep of the serie

        remove all freq above r_frq

        argument r_frq : the min frequency removed r_freq in Day -1

        dependence : scipy.fftpack fftfreq

        output: filtered np.array """

    freq = fftfreq(len(seriee), timestep/(3600*24))

    seriee[freq > r_frq] = 0

    return seriee


def plotinvfft(serie_filtered, data, name):
    """input: serie, data, name of the station

        use the inverse fast fourier transform then plot it

        dependence: matplotlib.pyplot scipy.fftpack ifft

        output: plot the wind-speed dependence over time"""

    inv_serie = ifft(serie_filtered)
    data = data[name].dropna()

    plt.title("filtered signal %s" % name)
    plt.xlabel(" Time")
    plt.ylabel(" Amplitude m/s")

    plt.plot(data.index, np.abs(inv_serie))
    plt.show()


