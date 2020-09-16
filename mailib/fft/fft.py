import pandas as pd
from toolbox import PolarToCartesian

from fft.fft_lib import *

if __name__ == '__main__':
    inpath = '../../fill/out/' 
    outpath = '../out/'
    #===========================================================================
    # Read data
    #===========================================================================
    df_sm = pd.read_csv(inpath + "Uw.csv", index_col=0, parse_dates=True)
    df_dm = pd.read_csv(inpath + "Vw.csv", index_col=0, parse_dates=True)
    df_T = pd.read_csv(inpath + "Ta.csv", index_col=0, parse_dates=True)

    index = df_sm.index[(df_sm.index.isin(df_dm.index))] # &(df_sm.index.isin(df_T.index))] 
    stanames =  df_sm.columns[(df_sm.columns.isin(df_dm.columns))] #&(df_sm.columns.isin(df_T.columns))]

    df_sm = df_sm.loc[index,stanames]
    df_dm = df_dm.loc[index,stanames]
    df_T = df_T.loc[index,stanames]

    df_u = []
    df_v = []
    for norm,theta in zip(df_sm.values, df_dm.values):
        U,V = PolarToCartesian(norm,theta)
        df_u.append(U)
        df_v.append(V)
        
#     df_T = pd.DataFrame(df_T, index=index, columns =stanames).resample('H').mean()   
    df_u = pd.DataFrame(df_u, index=index, columns =stanames).resample('H').mean()
    df_v = pd.DataFrame(df_v, index=index, columns =stanames).resample('H').mean()
    
    serie = df_T.loc[:,'C07']
    serie.dropna(inplace=True)
    index = serie.index
    timestep = gettimestep(serie.index)
    new_serie_fft = fft(serie)
    print new_serie_fft
    plotfft(new_serie_fft, timestep, 'C07', freq='d')
     
    fft_filtered = low_filter(new_serie_fft, timestep, r_frq=1.5) # Remove frequence under 0.9 days
    inv_serie = ifft(fft_filtered)
    plt.plot(index, np.abs(inv_serie))
    plt.show()
 
    fft_filtered = high_filter(new_serie_fft, timestep, r_frq=1.5) # Remove frequence under 0.9 days
    inv_serie = ifft(fft_filtered)
    plt.plot(index, np.abs(inv_serie))
    plt.show()
                           
    