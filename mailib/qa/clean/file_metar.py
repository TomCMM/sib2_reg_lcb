#===============================================================================
#     DESCRIPTION
#     this file contain the ipython history 
#===============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob


def correct_date(df):
    format = "%Y%m%d%H%M"
    years = df.iloc[:,0].apply(int).apply(str)
    months =df.iloc[:,1].apply(int).apply(str).str.rjust(2,'0')
    days = df.iloc[:,2].apply(int).apply(str).str.rjust(2,'0')
    hours =df.iloc[:,3].apply(int).apply(str).str.rjust(4,'0')


    dates = years+months+days+hours
    times = pd.to_datetime(dates, format=format)
    df_man = df
    df_man.set_index(times, inplace=True)
#         df_man.drop([0,1,2,3], axis=1, inplace=True)

    df_man[df_man == -999.0] = np.nan
    return df_man

if __name__ == '__main__':
#     inpath = "/home/thomas/phd/obs/staClim/metar/data/raw/data/data_all/"
    inpath= "/home/thomas/phd/obs/staClim/metar/data/clean_select/"
    outpath = "/home/thomas/phd/obs/staClim/metar/data/clean_select_byhand/"
    data_files = glob.glob(inpath + '*')

    dfs = []
    cnt = 0
    for file in data_files:
        try:
            df = pd.read_csv(file, index_col=1, skiprows=11, header=None)
            dfs.append(df)
        except ValueError:
            print "Could not read"
            print file
            cnt +=1

    dfs_concat = pd.concat(dfs, axis=0, join='outer')
    dfs_concat = dfs_concat.iloc[:,1:]
    columns = ['year','month','day','hour','id', 'lat','lon','alt','Sm m/s','Dm G','Ta C','Td C','SLP Pa', 'UR %', 'Visibilidade m', 'fenomeno', 'cloud cover fraction']
    variables =['Sm m/s', 'Dm G', 'Ta C', 'Td C', 'SLP Pa', 'Ua %', 'Visibilidade m', 'fenomeno', 'cloud cover fraction']
    metadata = ['id', 'lat', 'lon', 'alt']
    print dfs_concat
    dfs_rename = dfs_concat
    dfs_rename.columns = columns
    id_names = dfs_rename['id'].unique()


    for id_name in id_names:
        try:
            df = dfs_rename.loc[dfs_rename['id']==id_name, columns]
            df = correct_date(df)
            df.to_csv(outpath+id_name+'.csv')
        except:
            print dfs_rename
            print id_name

    data_files = glob.glob(outpath + '*')
#===============================================================================
# Convert knotw to m/s
#===============================================================================
    for f in data_files:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        new_df = df
        start_date ='2014-01-28'
        end_date = '2015-07-15'
        print df.index
        mask = (df.index < start_date) | (df.index >= end_date)
        # #df.loc[mask,'Sm m/s'].plot()
        # print mask
        index =  df.loc[mask, 'Sm m/s'].index
        # print index
        new_df.loc[mask,'Sm m/s'] = new_df.loc[mask, 'Sm m/s']*0.5144
#         new_df.loc[:,'Sm m/s'] = new_df.loc[:,'Sm m/s']*0.5144
        try:
            new_df.loc['2014-01-28','Sm m/s']= np.nan
            new_df.loc['2015-07-15','Sm m/s'] = np.nan

        except KeyError:
            pass
        new_df.to_csv(outpath+f[-8:])
#         df.loc[:,'Sm m/s'].plot()
#         new_df.loc[:,'Sm m/s'].plot()
#         plt.show()
