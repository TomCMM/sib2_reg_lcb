import pandas as pd
import glob
import os
if __name__ =='__main__':
    files_wind = glob.glob("/home/thomas/phd/obs/staClim/pcd_serradomar/data/merge_wind/*")
    files= glob.glob("/home/thomas/phd/obs/staClim/pcd_serradomar/data/merge_basevar/*")
    
    def read_pcd(file):
        df = pd.read_csv(file, parse_dates=True, index_col=0)
        df = df.rename(index=str, columns={"BarPres_Avg":"Pa H","RH_Avg":"Ua %", "AirTC_Avg":"Ta C" })
        df.index = pd.to_datetime(df.index, errors='coerce')  
        df.drop_duplicates(inplace=True) 
        df.index = pd.to_datetime(df.index, errors='coerce')  
        df = df.loc[~df.index.duplicated(keep='first')]
        df.index = pd.to_datetime(df.index, errors='coerce')  
        return df
    
    for f, fw in zip(files, files_wind):
        print "="*80
        df = read_pcd(f)
        df_wind = read_pcd(fw)
    
#         
        df = pd.concat([df.loc[:,['Ta C', 'Ua %', 'Pa H']], df_wind], axis=1, join='outer')
        df["TMP"] = df.index.values                # index is a DateTimeIndex
        df = df[df.TMP.notnull()]                  # remove all NaT values
        df.drop(["TMP"], axis=1, inplace=True)     # delete TMP again
        print df['Sm m/s']
        filename = os.path.basename(f)
        df.to_csv('/home/thomas/phd/obs/staClim/pcd_serradomar/data/merge/'+filename)
        





  