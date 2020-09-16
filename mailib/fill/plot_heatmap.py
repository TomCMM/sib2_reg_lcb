#===============================================================================
# DESCRIPTION
#    Plot the lookup table for the fill dataframe
#===============================================================================
import glob
import pandas as pd
from analysis.qa import outlier
from matplotlib.backends.backend_pdf import PdfPages



if __name__ == "__main__":
    vars = ['Ta C', 'Pa H','Qa g/kg','U m/s','V m/s']
    inpath = '../out/'
    outpath = "../res/"
    
    
    #    Serra Da Mantiquera
    Lat = [-25,-21]
    Lon = [-49, -43]
    params_selection={'Lat':Lat, 'Lon':Lon}
    
    
    for var in vars:
        print "Create quality control flag for the variable ->  " + str(var)
        f = glob.glob(inpath+ var[:2]+'*').pop()
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        
        df_available_data = df.groupby(lambda t: (t.year,t.month)).count().replace(0, np.nan)
        df_min = df.groupby(lambda t: (t.year,t.month)).min().replace(0, np.nan)#.subtract(df.groupby(lambda t: (t.year,t.month)).min().mean(axis=1), axis=0)
        df_mean = df.groupby(lambda t: (t.year,t.month)).mean().replace(0, np.nan)#.subtract(df.groupby(lambda t: (t.year,t.month)).mean().mean(axis=1), axis=0)
        df_max = df.groupby(lambda t: (t.year,t.month)).max().replace(0, np.nan)#.subtract(df.groupby(lambda t: (t.year,t.month)).max().mean(axis=1), axis=0)
        df_std = df.groupby(lambda t: (t.year,t.month)).aggregate(np.std).replace(0, np.nan)#.subtract(df.groupby(lambda t: (t.year,t.month)).aggregate(np.std).mean(axis=1), axis=0)
        df_outlier = df.groupby(lambda t: (t.year,t.month)).aggregate(outlier).replace(0, np.nan)
         
        with PdfPages(outpath + var[0:2] +'stats_per_month'+'.pdf') as pdf:
            print "Plotting ..."
            plot_heatmap_pdf(df_outlier,var, 'outlier per month',pdf)
            plot_heatmap_pdf(df_mean, var,'mean per month',pdf)
            plot_heatmap_pdf(df_std, var,'std per month',pdf)
            plot_heatmap_pdf(df_min,var, 'min per month',pdf)
            plot_heatmap_pdf(df_max,var, 'max per month',pdf)
            plot_heatmap_pdf(df_available_data,var, 'available data',pdf)
    
    