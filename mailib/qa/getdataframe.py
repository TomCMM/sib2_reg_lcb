from stadata.lib.LCBnet_lib import LCB_net, LCB_station, att_sta
import pandas as pd
import matplotlib.pyplot as plt

def create_dataframe(var, outpath, params_selection=None, From=None, To=None, check=True):
    """
    DESCRIPTION
        Create a daatframe for each climatic variables from the different stations network
    
    """

    net_sinda = LCB_net()
    net_inmet = LCB_net()
    net_iac = LCB_net()
    net_LCB = LCB_net()
    net_svg =  LCB_net()
    net_peg =  LCB_net()
    net_metar =  LCB_net()
    net_cetesb =  LCB_net()
    net_pcd =  LCB_net()

    Path_Sinda = '/home/thomas/PhD/obs-lcb/staClim/Sinda/obs_clean/Sinda/'
    Path_INMET ='/home/thomas/phd/obs/staClim/inmet/obs_clean/INMET/'
    Path_IAC ='/home/thomas/phd/obs/staClim/iac/obs_clean_byhand/IAC/'
    Path_LCB = '/home/thomas/phd/obs/lcbdata/obs/Merge/'
    Path_svg = '/home/thomas/phd/obs/staClim/svg/clean_byhand/SVG_2013_2016_Thomas_30m.csv'
    Path_peg = '/home/thomas/phd/obs/staClim/peg/Th_peg_tar30m.csv'
    Path_cetesb = "/home/thomas/phd/obs/staClim/cetesb/data/clean_byhand/"
    Path_metar = "/home/thomas/phd/obs/staClim/metar/data/clean_select_byhand/"
    Path_pcd = "/home/thomas/phd/obs/staClim/pcd_serradomar/data/clean/"
    
    metadatafile = '/home/thomas/phd/obs/staClim/metadata/database_metadata.csv'
    metadata_cetesb = "/home/thomas/phd/obs/staClim/cetesb/metadata/metadata_cetesb.csv"
    metadata_iac = "/home/thomas/phd/obs/staClim/iac/metadata/metadata_select_regional_wind.csv"
    metadata_metar = "/home/thomas/phd/obs/staClim/metar/metadata/metadata_select.csv"
    metadata_pcd = "/home/thomas/phd/obs/staClim/pcd_serradomar/metadata/metadata_pcd_serrra_do_mar.csv"

    AttSta_IAC = att_sta(metadata_iac)
    AttSta_Inmet = att_sta(metadatafile)
    AttSta_Sinda = att_sta(metadatafile)
    AttSta_LCB = att_sta(metadatafile)
    AttSta_cetesb = att_sta(metadata_cetesb)
    AttSta_metar = att_sta(metadata_metar)
    AttSta_pcd = att_sta(metadata_pcd)

    AttSta_IAC.setInPaths(Path_IAC)
    AttSta_Inmet.setInPaths(Path_INMET)
    AttSta_Sinda.setInPaths(Path_Sinda)
    AttSta_LCB.setInPaths(Path_LCB)
    AttSta_cetesb.setInPaths(Path_cetesb)
    AttSta_metar.setInPaths(Path_metar)
    AttSta_pcd.setInPaths(Path_pcd)

    stanames_IAC =  AttSta_IAC.stations(values=[]) # this does not work anymore
    stanames_Inmet = AttSta_Inmet.stations(values=['inmet'], params= params_selection)
    stanames_Sinda = AttSta_Sinda.stations(values=['sinda'],params= params_selection )
    stanames_LCB = AttSta_LCB.stations(values = ['ribeirao'], params= params_selection)
    stanames_cetesb = AttSta_cetesb.stations(values=[])
    stanames_metar = AttSta_metar.stations(values=[], params= params_selection)
    stanames_pcd = AttSta_pcd.stations(values=[], params= params_selection)

#------------------------------------------------------------------------------ 
# Create Dataframe
#------------------------------------------------------------------------------ 
    Files_IAC =AttSta_IAC.getatt(stanames_IAC,'InPath')
    Files_Inmet =AttSta_Inmet.getatt(stanames_Inmet,'InPath')
#     Files_Sinda =AttSta_Sinda.getatt(stanames_Sinda,'InPath')
    Files_LCB =AttSta_LCB.getatt(stanames_LCB,'InPath')
    Files_cetesb =AttSta_cetesb.getatt(stanames_cetesb,'InPath')
    Files_metar =AttSta_metar.getatt(stanames_metar,'InPath')
    Files_pcd =AttSta_pcd.getatt(stanames_pcd,'InPath')

#     net_sinda.AddFilesSta(Files_Sinda, net='Sinda')
    net_inmet.AddFilesSta(Files_Inmet, net='inmet')
    net_iac.AddFilesSta(Files_IAC, net='iac')
    net_LCB.AddFilesSta(Files_LCB)
    net_svg.AddFilesSta([Path_svg], net='svg')
    net_peg.AddFilesSta([Path_peg], net='peg')
    net_cetesb.AddFilesSta(Files_cetesb, net='cetesb')
    net_metar.AddFilesSta(Files_metar, net='metar')
    net_pcd.AddFilesSta(Files_pcd, net='pcd')

    df_iac = net_iac.getvarallsta(var=var,by='H',From=From, To = To)
    df_inmet = net_inmet.getvarallsta(var=var,by='H',From=From, To = To)
#     df_sinda = net_sinda.getvarallsta(var=var,by='3H',From=From, To = To)
    df_LCB = net_LCB.getvarallsta(var=var, by='H', From=From, To = To )
    df_svg = LCB_station(Path_svg, net='svg').getData(var=var, by='H', From=From, To = To )
    df_svg.columns =['svg']
    df_peg = LCB_station(Path_peg, net='peg').getData(var=var, by='H', From=From, To = To )
    df_peg.columns =['peg']
    df_metar = net_metar.getvarallsta(var=var, by='H', From=From, To = To )
    df_cetesb = net_cetesb.getvarallsta(var=var, by='H', From=From, To = To )
    df_pcd = net_pcd.getvarallsta(var=var, by='H', From=From, To = To )



    # check mean network
    if check:
        plt.plot(df_LCB.mean(1).groupby(lambda x:x.hour).mean(), label='LCB')
        plt.plot(df_iac.mean(1).groupby(lambda x:x.hour).mean(), label ='IAC')
        plt.plot(df_inmet.mean(1).groupby(lambda x:x.hour).mean(), label='inmet')
        plt.plot(df_metar.mean(1).groupby(lambda x:x.hour).mean(), label='metar')
        plt.plot(df_cetesb.mean(1).groupby(lambda x:x.hour).mean(), label='cetesb')
        plt.plot(df_svg.mean(1).groupby(lambda x:x.hour).mean(), label='svg')
        plt.plot(df_peg.mean(1).groupby(lambda x:x.hour).mean(), label='peg')
        plt.legend()
        plt.show()


    df = pd.concat([ df_LCB, df_iac, df_inmet, df_metar, df_cetesb, df_pcd], axis=1, join='outer')
    return df


if __name__ == '__main__':
    vars = [ 'Pa H','Sm m/s']
    outpath = "/home/thomas/phd/framework/qa/in/"

    #    Serra Da Mantiquera
    Lat = [-25,-21]
    Lon = [-49, -43]
    params_selection={'Lat':Lat, 'Lon':Lon}

    for var in vars:
        df = create_dataframe(var, outpath, params_selection, check=True)
        df.to_csv(outpath+var[:2]+'.csv')
