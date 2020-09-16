import pandas as pd


m_monica =  "/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/metadata_IAC.csv"
m_tom =  "/home/thomas/PhD/obs-lcb/staClim/IAC/stations_metadata.csv"

df_monica = pd.read_csv(m_monica)
df_tom = pd.read_csv(m_tom)

df1 = pd.merge(df_monica, df_tom, on ='ID', how='outer')

df_tom['Nome'] = df_tom['PCD']

df2 = pd.merge(df_monica, df_tom, on ='Nome', how='outer')



df1.to_csv('/home/thomas/metadata_IAC.csv')
df2.to_csv('/home/thomas/metadata2_IAC.csv')

for row in df[:]:
    print row
    
    
df = pd.read_csv("/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/metadata_IAC.csv",decimal=",",delimiter=",")


df['Lon'] = df['Lon'].str.replace(',', '.')
df['Lat'] = df['Lat'].str.replace(',', '.')

df['Lon'] = df['Lon'].astype(float)
df['Lat'] = df['Lat'].astype(float)
df['ID'] = df['ID'].astype(float)
df['ID'] = df['ID'].astype(int)
df['ID'] = df['ID'].astype(str)
df['ID']  =  df['Sigla'] + df['ID']
del df['Sigla']

df.to_csv('/home/thomas/metadataIAC.csv')
