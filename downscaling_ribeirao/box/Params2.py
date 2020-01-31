#===============================================================================
# DESCRIPTION
#    Perform a multi-linear regression and export the parameters
#===============================================================================


import pandas as pd
import numpy as np 
import statsmodels.api as sm

from matplotlib import pyplot as plt
#------------------------------------------------------------------------------ 
# Loading dataframe
df = pd.read_csv("/home/thomas/df_PCA_2.csv", index_col=1, parse_dates=True)
df = df.dropna(axis=0, how='any')
index_df = df.index
stations = df['stations']

print df
df['newindex'] = np.arange(len(df))
df.index = df['newindex'] 
del df['Unnamed: 0']
del df['newindex']
del df['stations']

#------------------------------------------------------------------------------ 
# split data table into data X and class labels y

esti_name = "Ua g/kg"
predi_name = ['CLWMR_500mb', 'HGT_950mb',"RH_950mb","TMP_950mb","VVEL_950mb", 'Alt','irr'] # Day 
# predi_name = ["TMP_950mb", 'Alt'] # Night


X = df[predi_name].values
y = df[esti_name].values

#===============================================================================
# Multiple regression
#===============================================================================


#===============================================================================
# Multiple regression of the component
#===============================================================================
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
ypred = est.predict(X)


def reform_df_stations(y_old, stations, index_df):
    y_old = pd.Series(y_old, index =range(len(ypred)) )
    stations = pd.Series(stations.values, index =range(len(stations)))
    newindex = pd.Series(index_df, index =range(len(index_df)))
    df_ypred = pd.concat([y_old, stations, newindex], axis=1)
    df_ypred.columns = ['ypred','stations', "newindex"]
    df_ypred = df_ypred.pivot(index='newindex', columns ='stations', values = 'ypred')
    return df_ypred

df_y = reform_df_stations(y, stations, index_df)
df_y_pred = reform_df_stations(ypred, stations, index_df)

df_y_pred[['C10','C09']].plot()
df_y[['C10','C09']].plot()

plt.plot(df_y[['C10','C04', 'C05', 'C06', 'C07', 'C08', 'C09']],'-', label = ['C10','C04', 'C05', 'C06', 'C07', 'C08', 'C09'])
plt.plot(df_y_pred[['C10','C04', 'C05', 'C06', 'C07', 'C08', 'C09']],'--', label = ['C10','C04', 'C05', 'C06', 'C07', 'C08', 'C09'])
plt.legend()
plt.show()



params = pd.DataFrame([est.params], columns = ['cst']+predi_name)
params.to_csv('/home/thomas/params.csv')
