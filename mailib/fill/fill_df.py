#===============================================================================
# DESCRIPTION
#    Fill the variables base on spatial regression with the two most correlated stations
#===============================================================================

import glob
from stadata.lib.LCBnet_lib import LCB_net, att_sta
# from fill.lib.fill import FillGap
import pandas as pd


if __name__ =='__main__':
    inpath = "/home/thomas/phd/framework/qa/out/qa_visual/"
    outpath = '../out/'
    outpath_summary_selected = '../res/'
    vars = ['Sm','Uw', 'Vw']
    nb_month_lim = 9 # must have at least 9 months of data to be use 

    for var in vars:
        print('Applying mask for the variable '+ str(var))
        
        
        # Select data for 2015 and remove stations with less than 9 months of data
        df = pd.read_csv(inpath+var+'.csv', index_col=0, parse_dates=True)
        df = df.loc['2015-01-01':'2015-12-31', :]
        df_available_data = df.groupby(lambda t: (t.year,t.month)).count().replace(0, np.nan)
        select_stanames = df_available_data.count()[df_available_data.count() > nb_month_lim].index
#         select_stanames = select_stanames[:20]
        df = df.loc[:, select_stanames]
        
        plot_heatmap(df_available_data.loc[:,select_stanames], var , outpath=outpath_summary_selected)
        
        df_filled = {}
        for i, response in enumerate(df.columns):
            print "filling station " + str(response) 
            print "remains [" + str(i) + "/" + str(len(df.columns)) + "]"
            predictors = list(df.columns)
            predictors.remove(response)
            selections = list(combinations(predictors, r=2)) # create all the possible combinations
            
            sorted_selection, params_selection = sort_predictors_by_corr(df, selections, verbose = False)
            df_filled[response] = filldf(df,response, sorted_selection, params_selection, verbose=False)
        df_filled = pd.DataFrame(df_filled)
        
        df_filled.to_csv(outpath + var+'.csv')
        print(df_filled)




