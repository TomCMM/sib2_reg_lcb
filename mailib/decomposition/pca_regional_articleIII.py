#===============================================================================
# DESCRIPTION
#    use the framework infrastructur to get the dataframe 
#===============================================================================

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from mailib.plot.map.map_stations import map_domain, add_stations_positions, add_wind_vector, add_loadings_as_marker
from model.statmod.statmod_lib import StaMod
from stadata.lib.LCBnet_lib import att_sta
from toolbox import is_point_in_domain

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["figure.figsize"] = (10,14)
matplotlib.rcParams.update({'font.size': 16})

matplotlib.rc('text', usetex=True)

def plot_daily_map_pc(pc_var, pc_u, pc_v , wind_stalatlon, var_stalatlon, raster_lat, raster_lon, raster_val, nb_pc=None, name=None, outpath="../../res/regional/combined/"):
    """"
    nb_pc: the pc number
    pc_var, pc_v, pc_u: reconstructed principal component values
    wind_stalatlon: Position of the station with wind measurement
    var_stalatlon: Position of the station with the variable of interest
    raster_lon, raster_lat: array with the latitude longitude of the background raster values "raster_val"

    """

#     stalatlon = AttSta.attributes.loc[stanames, ['Lat','Lon','network']]
#     print stalatlon

    try:
        idx = pc_var.index
    except AttributeError:
        try:
            idx = pc_v.index
        except AttributeError:
            idx = pc_u.index

    plt.close('all')
    for i, hour in enumerate(idx):
        PLT, map = map_domain(raster_lat, raster_lon, raster_val)

        try:
            map = add_loadings_as_marker(map, pc_var.loc[hour,:], var_stalatlon)
        except AttributeError:
            print('No value loadings to plot')
            pass

        try:
            map = add_wind_vector(map, pc_u.loc[hour,:], pc_v.loc[hour,:], wind_stalatlon)
        except AttributeError:
            print('No vector loadings to plot')
            pass
#         map = add_stations_positions(map,stalatlon)
         
        plt.title(str(hour)+" LT", fontsize=20)
        plt.legend(loc='best', framealpha=0.4)

        try:
            outfilename = outpath +'/PC'+str(nb_pc)+"/"+'PC'+str(nb_pc)+str(i).rjust(2,'0')+name+ "map_wind__regional.png"
            print outfilename
            plt.savefig(outfilename )
        except:
            outfilename = outpath +'/PC'+str(nb_pc)+"/"+name+'/'+'PC'+str(nb_pc)+str(i).rjust(2,'0')+name+ "map_wind__regional.png"
            print outfilename
            plt.savefig(outfilename )

        plt.close('all')

def plot_map_loadings(pc_var, pc_u, pc_v,  wind_stalatlon, var_stalatlon, raster_lat, raster_lon, raster_val, nb_pc=None,
                      name=None, outpath=None, pc_var2=None, use_pcvar2=False, stalatlon2=None, ax1=None, ribeirao=True,cs=None,
                      ax3=None, llcrnrlon=None,urcrnrlon=None,llcrnrlat=None, urcrnrlat=None,colorvector='k', alphavector=None, ykey=1.05,colorbaraltitude=False, vmin=None, vmax=None,
                      scale=1.25,scale_quiverkey=0.1,quiver_legend=r'$0.1$',linewidth=0.01, width=0.01):



    #     for i, hour in enumerate(pc_reconstruct_daily.index):

    # fig = matplotlib.pyplot.figure()


    # if use_pcvar2:
    #     fig, ((ax1, ax2), (ax3, ax4)) = matplotlib.pyplot.subplots(nrows=2, ncols=2)
    # else:
    #     fig, (ax1, ax3) = matplotlib.pyplot.subplots(nrows=2, ncols=1)

    # ax1.set_title("Regional domain")

    if llcrnrlon == None:
        llcrnrlon = -49
        urcrnrlon = -43
        llcrnrlat = -25
        urcrnrlat = -21

    # pc_var_regional = pc_var.drop([''])

    if not ribeirao:

        clevs = np.linspace(200, 1800, 17)
        plt_regional, map_regional, cs_regional = map_domain(raster_lat, raster_lon, raster_val, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, ax=ax1, clevs=clevs)

        #
        # parallels = np.arange(llcrnrlat, -urcrnrlat,1)
        # map_regional.drawparallels(parallels,labels=[1,0,0,0],fontsize=9, linewidth=0.2)
        # # draw meridians
        # meridians = np.arange(llcrnrlon, urcrnrlon,1)
        # map_regional.drawmeridians(meridians,labels=[0,0,0,1],fontsize=9, linewidth=0.2)

        # map_regional.drawmapscale(urcrnrlon-0.5, llcrnrlat+0.25, urcrnrlon+0.5, llcrnrlat+0.5, 50, barstyle='fancy',fontsize=11,units='km')

        try:
            map_regional, sc_loadings = add_loadings_as_marker(map_regional, pc_var, var_stalatlon, vmin=vmin, vmax=vmax)
            cs  = map_regional.colorbar(sc_loadings,location='bottom')
            # cs.ax.xaxis.set_ticks_position('top')
            # cs.ax.xaxis.set_rotation(45)
            cs.ax.set_xticklabels(cs.ax.get_xticklabels(), rotation=45)
            cs.ax.xaxis.set_ticks_position('bottom')
            cs.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        except AttributeError:
            print("No loadings to plot")
            pass

        try:
            map_regional = add_wind_vector(map_regional, pc_u, pc_v, wind_stalatlon,colorvector=colorvector, alphavector=alphavector,
                                           ykey=ykey, scale=scale,scale_quiverkey=scale_quiverkey,quiver_legend=quiver_legend,  linewidth=linewidth, width=width)
        except AttributeError:
            print('No vector plot')
            pass

    if use_pcvar2:
        plt_regional, map_regional, cs_regional = map_domain(raster_lat, raster_lon, raster_val, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, ax=ax2)
        parallels = np.arange(llcrnrlat, -urcrnrlat,1)
        map_regional.drawparallels(parallels,labels=[1,0,0,0],fontsize=11, linewidth=0.2)
        # draw meridians
        meridians = np.arange(llcrnrlon, urcrnrlon,1)
        map_regional.drawmeridians(meridians,labels=[0,0,0,1],fontsize=11, linewidth=0.2)

        map_regional.drawmapscale(urcrnrlon-0.5, llcrnrlat+0.25, urcrnrlon+0.5, llcrnrlat+0.5, 50, barstyle='fancy',fontsize=11,units='km')
        try:
            map_regional, sc_loadings = add_loadings_as_marker(map_regional, pc_var2, stalatlon2)
            cs  = map_regional.colorbar(sc_loadings,location='top', pad=0.2)
            cs.ax.xaxis.set_ticks_position('top')
            cs.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        except AttributeError:
            print("No loadings to plot")
            pass

    raster_lat_posses = np.loadtxt("/home/thomas/phd/framework/predictors/in/topo/ribeirao/posseslatitude.txt", delimiter=',')
    raster_lon_posses = np.loadtxt("/home/thomas/phd/framework/predictors/in/topo/ribeirao/posseslongitude.txt", delimiter=',')
    raster_val_posses = np.loadtxt("/home/thomas/phd/framework/predictors/in/topo/ribeirao/posses@PERMANENT", delimiter=',')
    raster_lat_posses = raster_lat_posses[:,:-1]
    raster_lon_posses = raster_lon_posses[:,:-1]
    raster_val_posses = raster_val_posses[:,:-1]


    # ax2.set_title("Ribeirao Das Posses")

    if ribeirao:



        plt_ribeirao, map_ribeirao, cs_ribeirao = map_domain(raster_lat_posses, raster_lon_posses, raster_val_posses, ax=ax1, ribeirao=True)



        try:
            map_ribeirao, sc_loadings = add_loadings_as_marker(map_ribeirao, pc_var, var_stalatlon, vmin=vmin, vmax=vmax)
            map_ribeirao.colorbar(sc_loadings)
            # cs  = map_ribeirao.colorbar(sc_loadings, location='bottom')
            # cs.ax.xaxis.set_ticks_position('bottom')
            # cs.ax.xaxis.set_rotation(45)
            # cs.ax.set_xticklabels(cs.ax.get_xticklabels(), rotation=45)
            # cs.ax.xaxis.set_ticks_position('bottom', pad=0.1)
            # cs.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        except AttributeError:
            print('no loadings to plot')

        if colorbaraltitude:
            cbar = map_ribeirao.colorbar(cs_ribeirao, location='bottom')
            cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
            cbar.set_label('Altitude (m)')

        try:
            map_ribeirao = add_wind_vector(map_ribeirao, pc_u, pc_v, wind_stalatlon,colorvector=colorvector,alphavector=alphavector)
        except AttributeError:
            print('No vector plot')
            pass

        if use_pcvar2:

            plt_ribeirao, map_ribeirao, cs_ribeirao = map_domain(raster_lat_posses, raster_lon_posses, raster_val_posses, ax=ax4)

            try:
                map_ribeirao, sc_loadings = add_loadings_as_marker(map_ribeirao, pc_var2, stalatlon2)
            except AttributeError:
                print('no loadings to plot')

            cbar = map_ribeirao.colorbar(cs_ribeirao, location='bottom')
            cbar.set_label('Altitude (m)')


        # plt.title(" Loadings of the pc " + str(nb_pc), fontsize=20)
        # plt.legend(loc='best', framealpha=0.4)
        # cs  = map_ribeirao.colorbar(sc_loadings)
        #
        #
        #
        # llcrnrlon = -46.24
        # urcrnrlon = raster_lon_posses.max()
        # llcrnrlat = -22.89
        # urcrnrlat = -22.83
        #
        # parallels = np.arange(llcrnrlat, urcrnrlat,0.01)
        # map_ribeirao.drawparallels(parallels,labels=[1,0,0,0],fontsize=11, linewidth=0.2)
        # # draw meridians
        # meridians = np.arange(llcrnrlon, urcrnrlon,0.01)
        # map_ribeirao.drawmeridians(meridians,labels=[0,0,0,1],fontsize=11, linewidth=0.2)
        #


        # map_ribeirao.drawmapscale(urcrnrlon-0.5, llcrnrlat+0.25, urcrnrlon+0.5, llcrnrlat+0.5, 50, barstyle='fancy',fontsize=11,units='km')

    # plt.tight_layout()


    if not outpath:
        return ax1, ax3, cs
    else:
        outfilename = outpath +'PC' + str(nb_pc)+ name + "map_wind__regional.png"
        print outfilename
        plt.savefig(outfilename )
        # return map
        plt.close('all')

def do_regional_pca(df, nb_pc, outpath= None):

        #===============================================================================
        # Perform PCA
        #===============================================================================
    #     nb_pc = 3
    #
    #     pca = PCA(nb_pc)
    #     pca.fit(df)
    #     scores = pd.DataFrame(pca.transform(df), columns= ['PC' for pc in range(nb_pc)], index= df.index)
    #     loadings = pca.components_
    #     expvar = pca.explained_variance_
    #
    #     plt.bar(range(len(expvar)), expvar)
    #     plt.show()
    #
    #     plt.plot(scores)
    #     plt.show()

        #===============================================================================
        # With local_downscaling
        #===============================================================================

        AttSta = att_sta()
        stamod = StaMod(df, AttSta)
        stamod.pca_transform(nb_PC=nb_pc, cov=True, standard=False, remove_mean1 =False, remove_mean0 =False)
        daily_scores = stamod.scores.groupby(lambda t: (t.hour)).mean()
        loadings = stamod.eigenvectors
        scores = stamod.scores

        # Basic plot components
        # plt.figure()
        # ax_daily = daily_scores.plot()
        # # plt.savefig(outpath +'daily_scores.png' )
        # # plt.close()
        # ax_scores = stamod.plot_scores_ts(output = outpath+"scores.png")
        # stamod.plot_exp_var(output = outpath +'exp_var' )

        dfs_pcs_reconstruct = stamod.pca_reconstruct() # reconstruction

        return stamod, loadings, scores ,dfs_pcs_reconstruct

def get_dataframes():
    #===========================================================================
    # Prepare input dataset
    #===========================================================================
    # from the filled data
    df_u = pd.read_csv("/home/thomas/phd/framework/fill/out/Uw.csv", index_col=0, parse_dates=True).resample('H').mean()
    df_v = pd.read_csv("/home/thomas/phd/framework/fill/out/Vw.csv", index_col=0, parse_dates=True).resample('H').mean()
    df_q = pd.read_csv("/home/thomas/phd/framework/fill/out/Qa.csv", index_col=0, parse_dates=True).resample('H').mean()
    df_t = pd.read_csv("/home/thomas/phd/framework/fill/out/Ta.csv", index_col=0, parse_dates=True).resample('H').mean()

# #     # from the high filter data
#     df_u = pd.read_csv("/home/thomas/phd/framework/fft/out/high/Uw.csv", index_col=0, parse_dates=True).resample('H').mean()
#     df_v = pd.read_csv("/home/thomas/phd/framework/fft/out/high/Vw.csv", index_col=0, parse_dates=True).resample('H').mean()
#     df_q = pd.read_csv("/home/thomas/phd/framework/fft/out/high/Qa.csv", index_col=0, parse_dates=True).resample('H').mean()
#     df_t = pd.read_csv("/home/thomas/phd/framework/fft/out/high/Ta.csv", index_col=0, parse_dates=True).resample('H').mean()

#     # from the low filter data
#     df_u = pd.read_csv("/home/thomas/phd/framework/fft/out/low/Uw.csv", index_col=0, parse_dates=True).resample('H').mean()
#     df_v = pd.read_csv("/home/thomas/phd/framework/fft/out/low/Vw.csv", index_col=0, parse_dates=True).resample('H').mean()
#     df_q = pd.read_csv("/home/thomas/phd/framework/fft/out/low/Qa.csv", index_col=0, parse_dates=True).resample('H').mean()
#     df_t = pd.read_csv("/home/thomas/phd/framework/fft/out/low/Ta.csv", index_col=0, parse_dates=True).resample('H').mean()

    # Remove some stations in the Ribeirao Das Posses
    # df_t.drop([ 'C08','C06', 'C04','C12','C13'], inplace=True, axis=1)
    # df_q.drop([ 'C08','C06', 'C04','C12','C13'], inplace=True, axis=1)
    # df_u.drop([ 'C08','C06', 'C04','C12','C13'], inplace=True, axis=1)
    # df_v.drop([ 'C08','C06', 'C04','C12','C13'], inplace=True, axis=1)

    # select sta in domain
    AttSta = att_sta("/home/thomas/phd/obs/staClim/metadata/database_metadata.csv")
    raster_lon = np.loadtxt("/home/thomas/phd/framework/predictors/in/topo/regional/4943_2125_lowreslongitude.txt", delimiter=',')
    raster_lat = np.loadtxt("/home/thomas/phd/framework/predictors/in/topo/regional/4943_2125_lowreslatitude.txt", delimiter=',')


    wind_stalatlon = AttSta.attributes.loc[df_u.columns, ['Lat','Lon','network']]
    T_stalatlon = AttSta.attributes.loc[df_t.columns, ['Lat','Lon','network']]
    Q_stalatlon = AttSta.attributes.loc[df_q.columns, ['Lat','Lon','network']]

    T_stalatlon_indomain_lat, T_stalatlon_indomain_lon =  is_point_in_domain(raster_lat, raster_lon, T_stalatlon.loc[:,'Lat'], T_stalatlon.loc[:,'Lon'])
    Q_stalatlon_indomain_lat, Q_stalatlon_indomain_lon =  is_point_in_domain(raster_lat, raster_lon, Q_stalatlon.loc[:,'Lat'], Q_stalatlon.loc[:,'Lon'])
    wind_stalatlon_indomain_lat, wind_stalatlon_indomain_lon =  is_point_in_domain(raster_lat, raster_lon, wind_stalatlon.loc[:,'Lat'], wind_stalatlon.loc[:,'Lon'])


    df_u = df_u.loc[:,wind_stalatlon_indomain_lat.index]
    df_v = df_v.loc[:,wind_stalatlon_indomain_lat.index]
    df_t = df_t.loc[:,T_stalatlon_indomain_lat.index]
    df_q = df_q.loc[:,Q_stalatlon_indomain_lat.index]

    df_t.drop([ 'C08','C06', 'C04','C12','C13', 'C15'], inplace=True, axis=1)
    df_q.drop([ 'C08','C06' ,'C04','C12','C13', 'C15'], inplace=True, axis=1)
    df_u.drop([ 'C08','C06' ,'C04','C12','C13', 'C15','A715', 'A736' ], inplace=True, axis=1)
    df_v.drop(['C08','C06' ,'C04','C12','C13', 'C15','A715', 'A736'], inplace=True, axis=1)



    # df_u_forquv = df_u
    # df_v_forquv = df_v

    # Rename columns
    staname_wind = df_u.columns
    staname_q = df_q.columns
    staname_t = df_t.columns

    # staname_wind_forquv = staname_wind.drop(['SBES', u'SBGL', u'SBGR', u'SBGW', u'SBJR',u'SBKP', u'SBPC', u'SBRP', u'SBSC', u'SBSJ', u'SBSP', u'SBST', u'SBYS', u'A514',
    #                                          'A529', 'A609','A714','A726','dv136','pr141', 'SBBQ','C06','C10','A509','A515', 'A531', 'A619'])
    #

    # df_u_forquv = df_u_forquv.loc[:,staname_wind_forquv]
    # df_v_forquv = df_v_forquv.loc[:,staname_wind_forquv]

    df_t.columns = ['T_'+str(sta) for sta in staname_t]
    df_q.columns = ['Q_'+str(sta) for sta in staname_q]
    df_u.columns = ['U_'+str(sta) for sta in staname_wind]
    df_v.columns =  ['V_'+str(sta) for sta in staname_wind]
    # df_u_forquv.columns =  ['U_'+str(sta) for sta in df_u_forquv.columns]
    # df_v_forquv.columns =  ['V_'+str(sta) for sta in df_v_forquv.columns]

    # Preprocessing


    df_t_old = df_t
    df_q_old = df_q
    df_u_old = df_u
    df_v_old = df_v

    df_t_mean = df_t.mean(axis=0)
    df_q_mean = df_q.mean(axis=0)
    df_u_mean = df_u.mean(axis=0)
    df_v_mean = df_v.mean(axis=0)

    df_u = df_u - df_u_mean
    df_v = df_v - df_v_mean
    df_t = df_t - df_t_mean
    df_q = df_q - df_q_mean
    # df_u_forquv = df_u_forquv - df_u_forquv.mean(axis=0)
    # df_v_forquv = df_v_forquv - df_v_forquv.mean(axis=0)
    #
    # from sklearn.preprocessing import StandardScaler
    # scaler_t = StandardScaler()
    # scaler_q = StandardScaler()
    # scaler_u = StandardScaler()
    # scaler_v = StandardScaler()
    #
    # df_t = pd.DataFrame(scaler_t.fit_transform(df_t), index=df_t.index, columns=df_t.columns)
    # df_q = pd.DataFrame(scaler_q.fit_transform(df_q), index=df_q.index, columns=df_q.columns)
    # df_u = pd.DataFrame(scaler_u.fit_transform(df_u), index=df_u.index, columns=df_u.columns)
    # df_v = pd.DataFrame(scaler_v.fit_transform(df_v), index=df_v.index, columns=df_v.columns)


    # Normalize
    # df_t = apply_func(df_t, normalize)
    # df_q = apply_func(df_q, normalize)
    # df_u = apply_func(df_u, normalize)
    # df_v = apply_func(df_v, normalize)

    # df_u_forquv = apply_func(df_u_forquv, normalize)
    # df_v_forquv = apply_func(df_v_forquv, normalize)


    def norm(row):
        w = np.sqrt(np.sum(row**2))
        x_norm2 = row/w
        return x_norm2

    def W(row):
        w = np.sqrt(np.sum(row**2))
        x_norm2 = row/w
        return w

    print('normalized weight')
    df_t_w  = df_t.apply(W, axis=1)
    df_q_w  = df_q.apply(W, axis=1)
    df_u_w  = df_u.apply(W, axis=1)
    df_v_w  = df_v.apply(W, axis=1)

    df_t  = df_t.apply(norm, axis=1)
    df_q  = df_q.apply(norm, axis=1)
    df_u  = df_u.apply(norm, axis=1)
    df_v  = df_v.apply(norm, axis=1)


    # #Using Sklearn
    # normalizer_x = Normalizer(norm = "l2").fit(x)
    # x_norm = normalizer_x.transform(x)
    # print x_norm

    # Manually
    # w = np.sqrt(np.sum(x**2))
    # x_norm2 = x/w
    # print x_norm2

    # Create combined dataframe
    df_tuv = pd.concat([df_t, df_u, df_v], join='inner', axis=1)
    df_quv = pd.concat([df_q, df_u, df_v], join='inner', axis=1)
    df_tquv = pd.concat([df_t,df_q, df_u, df_v], join='inner', axis=1)
    df_uv = pd.concat([df_u, df_v], join='inner', axis=1)

    return df_t, df_q,df_u, df_v, df_tuv,df_quv,  df_tquv, df_uv, staname_t, staname_q, staname_wind, \
           df_t_mean, df_q_mean,df_u_mean, df_v_mean, df_t_w, df_q_w,df_u_w, df_v_w, df_t_old, df_q_old,df_u_old, df_v_old

def get_background_and_pos(staname_t, staname_q, staname_wind):

    #===========================================================================
    # Plot daily wind vector over the region
    #===========================================================================
    raster_val = np.loadtxt("/home/thomas/phd/framework/predictors/in/topo/regional/4943_2125_lowres@PERMANENT", delimiter=',')
    raster_lon = np.loadtxt("/home/thomas/phd/framework/predictors/in/topo/regional/4943_2125_lowreslongitude.txt", delimiter=',')
    raster_lat = np.loadtxt("/home/thomas/phd/framework/predictors/in/topo/regional/4943_2125_lowreslatitude.txt", delimiter=',')

    AttSta = att_sta("/home/thomas/phd/obs/staClim/metadata/database_metadata.csv")
    wind_stalatlon = AttSta.attributes.loc[staname_wind, ['Lat','Lon','network']]
    T_stalatlon = AttSta.attributes.loc[staname_t, ['Lat','Lon','network']]
    Q_stalatlon = AttSta.attributes.loc[staname_q, ['Lat','Lon','network']]






    return raster_val, raster_lon, raster_lat, wind_stalatlon,  T_stalatlon, Q_stalatlon

def plot_fig2_T():
    gs = gridspec.GridSpec(7, 2)
    ax1 = plt.subplot(gs[0:2, 0])
    ax2 = plt.subplot(gs[0:2, 1])
    ax3 = plt.subplot(gs[2:4, 0])
    ax4 = plt.subplot(gs[2:4, 1])

    ax5 = plt.subplot(gs[4, :])
    ax6 = plt.subplot(gs[5, :])
    ax7 = plt.subplot(gs[6, :])


    ax_reg, ax_rib,cs1 = plot_map_loadings(loadings_t.loc[1, :].filter(regex='T_'), None, None,
                          None, T_stalatlon, raster_lat,
                          raster_lon, raster_val, nb_pc=1, name='T',outpath=None, ax1=ax1, ribeirao=False)

    ax_reg, ax_rib,cs2 = plot_map_loadings(loadings_t.loc[2, :].filter(regex='T_'), None, None,
                          None, T_stalatlon, raster_lat,
                          raster_lon, raster_val, nb_pc=2, name='T',outpath=None, ax1=ax2, ribeirao=False)

    ax_reg, ax_rib,cs3 = plot_map_loadings(loadings_t.loc[3, :].filter(regex='T_'), None, None,
                          None, T_stalatlon, raster_lat,
                          raster_lon, raster_val, nb_pc=3, name='T',outpath=None, ax1=ax3, ribeirao=False)



    daily_scores =scores_t.groupby(lambda t: (t.hour)).mean()
    daily_scores.columns = ['1','2', '3']
    daily_scores.plot(ax=ax4, color=['b','r','k'], linewidth=3)
    ax4.set_xlabel('Hours')
    ax4.set_ylabel('PC scores')


    scores_t.iloc[:, 0].plot(ax=ax5, color='b', linewidth=0.5)
    scores_t.iloc[:,1].plot(ax=ax6, color='r', linewidth=0.5)
    scores_t.iloc[:,2].plot(ax=ax7, color='k', linewidth=0.5)

    ax1.set_ylabel('PC1 loadings', labelpad=25)
    ax2.set_ylabel('PC2 loadings', labelpad=25)
    ax3.set_ylabel('PC3 loadings', labelpad=25)

    ax5.set_ylabel('PC1 scores')
    ax6.set_ylabel('PC2 scores')
    ax7.set_ylabel('PC3 scores')
    ax7.set_xlabel('Date')

    ax4.axhline(0,c='k')
    ax5.axhline(0,c='k')
    ax6.axhline(0,c='k')
    ax7.axhline(0,c='k')

    for t, ax in zip(["a)","b)",'c)','d)'],[ax1, ax2,ax3, ax4]):
        ax.text(-0.10, 1.2, t, transform=ax.transAxes, size=14, weight='bold')

    for t, ax in zip(['e)', 'f)', 'g)','h)','i)','j)'],[ax5, ax6, ax7]):
        ax.text(-0.05, 1.0, t, transform=ax.transAxes, size=14, weight='bold')

    ax5.set_xticks([], [])
    ax6.set_xticks([], [])
    # ax7.set_xticks([], [])

    # plt.savefig(outpath +'daily_scores.png' )
    # plt.close()
    # ax_scores = stamod.plot_scores_ts(output = outpath+"scores.png")
    # stamod.plot_exp_var(output = outpath +'exp_var' )
    plt.tight_layout()
    plt.savefig('/home/thomas/phd/framework/res/regional/chapIII/Fig2_T.png', dpi=300 )
    plt.close('all')

def plot_fig3_Q():
    gs = gridspec.GridSpec(7, 2)
    ax1 = plt.subplot(gs[0:2, 0])
    ax2 = plt.subplot(gs[0:2, 1])
    ax3 = plt.subplot(gs[2:4, 0])
    ax4 = plt.subplot(gs[2:4, 1])

    ax5 = plt.subplot(gs[4, :])
    ax6 = plt.subplot(gs[5, :])
    ax7 = plt.subplot(gs[6, :])


    ax_reg, ax_rib,cs1 = plot_map_loadings(loadings_q.loc[1, :].filter(regex='Q_'), None, None,
                          None, Q_stalatlon, raster_lat,
                          raster_lon, raster_val, nb_pc=1, name='Q',outpath=None, ax1=ax1, ax3 = None, ribeirao=False)

    ax_reg, ax_rib,cs2 = plot_map_loadings(loadings_q.loc[2, :].filter(regex='Q_'), None, None,
                          None, Q_stalatlon, raster_lat,
                          raster_lon, raster_val, nb_pc=2, name='Q',outpath=None, ax1=ax2, ax3 = None, ribeirao=False)

    ax_reg, ax_rib,cs3 = plot_map_loadings(loadings_q.loc[3, :].filter(regex='Q_'), None, None,
                          None, Q_stalatlon, raster_lat,
                          raster_lon, raster_val, nb_pc=3, name='Q',outpath=None, ax1=ax3, ax3 = None, ribeirao=False)


    daily_scores =scores_q.groupby(lambda t: (t.hour)).mean()
    daily_scores.columns = ['1','2', '3']
    daily_scores.plot(ax=ax4, color=['b','r','k'], linewidth=3)
    ax4.set_xlabel('Hours')
    ax4.set_ylabel('PC scores')


    scores_q.iloc[:, 0].plot(ax=ax5,color='b', linewidth=0.5)
    scores_q.iloc[:,1].plot(ax=ax6, color='r', linewidth=0.5)
    scores_q.iloc[:,2].plot(ax=ax7, color='k', linewidth=0.5)
    ax1.set_ylabel('PC1 loadings', labelpad=25)
    ax2.set_ylabel('PC2 loadings', labelpad=25)
    ax3.set_ylabel('PC3 loadings', labelpad=25)

    ax5.set_ylabel('PC1 scores')
    ax6.set_ylabel('PC2 scores')
    ax7.set_ylabel('PC3 scores')
    ax7.set_xlabel('Date')

    ax4.axhline(0,c='k')
    ax5.axhline(0,c='k')
    ax6.axhline(0,c='k')
    ax7.axhline(0,c='k')

    for t, ax in zip(["a)","b)",'c)','d)'],[ax1, ax2,ax3, ax4]):
        ax.text(-0.10, 1.2, t, transform=ax.transAxes, size=14, weight='bold')

    for t, ax in zip(['e)', 'f)', 'g)','h)','i)','j)'],[ax5, ax6, ax7]):
        ax.text(-0.05, 1.0, t, transform=ax.transAxes, size=14, weight='bold')


    # plt.savefig(outpath +'daily_scores.png' )
    # plt.close()
    # ax_scores = stamod.plot_scores_ts(output = outpath+"scores.png")
    # stamod.plot_exp_var(output = outpath +'exp_var' )
    plt.tight_layout()
    plt.savefig('/home/thomas/phd/framework/res/regional/chapIII/Fig3_Q.png', dpi=300 )
    plt.close('all')

def plot_fig4_UV():
    gs = gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    ax5 = plt.subplot(gs[2, 0])
    # ax6 = plt.subplot(gs[2, 1])

    # llcrnrlon = -49.25
    # urcrnrlon = -42.75
    # llcrnrlat = -25.25
    # urcrnrlat = -20.75

    ax_reg, ax_rib,cs1 =plot_map_loadings(None, loadings_uv.loc[1, :].filter(regex='U_'), loadings_uv.loc[1, :].filter(regex='V_'),
                      wind_stalatlon, None, raster_lat,
                      raster_lon, raster_val, nb_pc=1, name='UV',outpath=None, ax1=ax1, ax3 = None, ribeirao=False,colorvector='b')

    ax_reg, ax_rib,cs2 =plot_map_loadings(None, loadings_uv.loc[2, :].filter(regex='U_'), loadings_uv.loc[2, :].filter(regex='V_'),
                      wind_stalatlon, None, raster_lat,
                      raster_lon, raster_val, nb_pc=2, name='UV',outpath=None, ax1=ax2, ax3 = None, ribeirao=False,colorvector='b')

    ax_reg, ax_rib,cs3 =plot_map_loadings(None, loadings_uv.loc[3, :].filter(regex='U_'), loadings_uv.loc[3, :].filter(regex='V_'),
                      wind_stalatlon, None, raster_lat,
                      raster_lon, raster_val, nb_pc=3, name='UV',outpath=None, ax1=ax3, ax3 = None, ribeirao=False,colorvector='b')

    ax_reg, ax_rib,cs4 =plot_map_loadings(None, loadings_uv.loc[4, :].filter(regex='U_'), loadings_uv.loc[4, :].filter(regex='V_'),
                      wind_stalatlon, None, raster_lat,
                      raster_lon, raster_val, nb_pc=4, name='UV',outpath=None, ax1=ax4, ax3 = None, ribeirao=False,colorvector='b')

    # ax_reg, ax_rib,cs5 =plot_map_loadings(None, loadings_uv.loc[5, :].filter(regex='U_'), loadings_uv.loc[5, :].filter(regex='V_'),
    #                   wind_stalatlon, None, raster_lat,
    #                   raster_lon, raster_val, nb_pc=4, name='UV',outpath=None, ax1=ax5, ax3 = None, ribeirao=False)
    #


    daily_scores =scores_uv.groupby(lambda t: (t.hour)).mean()
    daily_scores.columns = ['1','2', '3','4']
    daily_scores.plot(ax=ax5, color=['b','r','k', 'limegreen'], linewidth=3)
    ax5.set_xlabel('Hours')
    ax5.set_ylabel('PC scores')
    ax5.axhline(0,c='k')

    ax1.set_title('PC1 vector loadings')
    ax2.set_title('PC2 vector loadings')
    ax3.set_title('PC3 vector loadings')
    ax4.set_title('PC4 vector loadings')

    # ax2.set_yticks([], [])
    # ax4.set_yticks([], [])

    for t, ax in zip(["a)","b)",'c)','d)','e)', 'f)', 'g)','h)','i)','j)'],[ax1, ax2,ax3, ax4, ax5]):
        ax.text(-0.10, 1.10, t, transform=ax.transAxes, size=14, weight='bold')


    plt.tight_layout(pad=2)
    plt.savefig('/home/thomas/phd/framework/res/regional/chapIII/Fig4_UV.png', dpi=300 )
    # plt.show()
    plt.close('all')

def plot_fig5_TUV_QUV():
    gs = gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    ax5 = plt.subplot(gs[2, 0])
    ax6 = plt.subplot(gs[2, 1])

    ax_reg, ax_rib,cs1 =plot_map_loadings(loadings_tuv.loc[2, :].filter(regex='T_'), loadings_tuv.loc[2, :].filter(regex='U_'), loadings_tuv.loc[2, :].filter(regex='V_'),
                      wind_stalatlon, T_stalatlon, raster_lat,
                      raster_lon, raster_val, nb_pc=2, name='UV',outpath=None, ax1=ax1, ribeirao=False, alphavector=0.5, colorvector='b', ykey=1.25)

    ax_reg, ax_rib,cs2 =plot_map_loadings(loadings_quv.loc[2, :].filter(regex='Q_'), loadings_quv.loc[2, :].filter(regex='U_'), loadings_quv.loc[2, :].filter(regex='V_'),
                  wind_stalatlon, Q_stalatlon, raster_lat,
                  raster_lon, raster_val, nb_pc=2, name='UV',outpath=None, ax1=ax2, ribeirao=False, alphavector=0.5, colorvector='b', ykey=1.25)

    ax_reg, ax_rib,cs3 =plot_map_loadings(loadings_tuv.loc[4, :].filter(regex='T_'), loadings_tuv.loc[4, :].filter(regex='U_'), loadings_tuv.loc[4, :].filter(regex='V_'),
                      wind_stalatlon, T_stalatlon, raster_lat,
                      raster_lon, raster_val, nb_pc=4, name='UV',outpath=None, ax1=ax3, ribeirao=False, alphavector=0.5, colorvector='b', ykey=1.25)

    ax_reg, ax_rib,cs4 =plot_map_loadings(loadings_quv.loc[4, :].filter(regex='Q_'), loadings_quv.loc[4, :].filter(regex='U_'), loadings_quv.loc[4, :].filter(regex='V_'),
                  wind_stalatlon, Q_stalatlon, raster_lat,
                  raster_lon, raster_val, nb_pc=4, name='UV',outpath=None, ax1=ax4, ribeirao=False, alphavector=0.5, colorvector='b', ykey=1.25)

    daily_scores =scores_tuv.loc[:,[2,4]].groupby(lambda t: (t.hour)).mean()
    daily_scores.plot(ax=ax5, color=['r','limegreen'], linewidth=3)
    ax5.set_xlabel('Hours')
    ax5.set_ylabel('PC scores')
    ax5.axhline(0,c='k')

    daily_scores =scores_quv.loc[:,[2,4]].groupby(lambda t: (t.hour)).mean()
    daily_scores.plot(ax=ax6,  color=['r','limegreen'], linewidth=3)
    ax6.set_xlabel('Hours')
    ax6.set_ylabel('PC scores')
    ax6.axhline(0,c='k')

    for t, ax in zip(["a)","b)",'c)','d)','e)', 'f)', 'g)','h)','i)','j)'],[ax1, ax2,ax3, ax4, ax5, ax6]):
        ax.text(-0.10, 1.10, t, transform=ax.transAxes, size=14, weight='bold')

    ax1.set_ylabel('PC2 loadings', labelpad=30)
    ax3.set_ylabel('PC4 loadings', labelpad=30)
    #
    cs1.set_label('Temperature and wind', labelpad=-50)
    cs2.set_label('Specific humidity and wind', labelpad=-50)

    # plt.tight_layout(pad=2.5)
    plt.tight_layout()
    plt.savefig('/home/thomas/phd/framework/res/regional/chapIII/Fig5_TUV_QUV.png', dpi=300 )
    # plt.show()
    plt.close('all')

def plot_fig6_ribeirao():
    gs = gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    ax5 = plt.subplot(gs[2, 0])
    # ax6 = plt.subplot(gs[2, 1])


    ax_reg, ax_rib,cs3 =plot_map_loadings(None, loadings_uv.loc[3, :].filter(regex='U_'), loadings_uv.loc[3, :].filter(regex='V_'),
                      wind_stalatlon, None, raster_lat,
                      raster_lon, raster_val, nb_pc=3, name='UV',outpath=None, ax1=ax1, ribeirao=True, colorvector='b',  alphavector=0.5)

    ax_reg, ax_rib,cs4 =plot_map_loadings(None, loadings_uv.loc[4, :].filter(regex='U_'), loadings_uv.loc[4, :].filter(regex='V_'),
                      wind_stalatlon, None, raster_lat,
                      raster_lon, raster_val, nb_pc=4, name='UV',outpath=None, ax1=ax2, ribeirao=True, colorvector='b' ,alphavector=0.5)


    ax_reg, ax_rib,cs2 =plot_map_loadings(loadings_tuv.loc[4, :].filter(regex='T_'), loadings_tuv.loc[4, :].filter(regex='U_'), loadings_tuv.loc[4, :].filter(regex='V_'),
                  wind_stalatlon, T_stalatlon, raster_lat,
                  raster_lon, raster_val, nb_pc=2, name='UV',outpath=None, ax1=ax3, ribeirao=True, colorvector='b',alphavector=0.5)

    ax_reg, ax_rib,cs2 =plot_map_loadings(loadings_quv.loc[4, :].filter(regex='Q_'), loadings_quv.loc[4, :].filter(regex='U_'), loadings_quv.loc[4, :].filter(regex='V_'),
                  wind_stalatlon, Q_stalatlon, raster_lat,
                  raster_lon, raster_val, nb_pc=2, name='UV',outpath=None, ax1=ax4, ribeirao=True, colorvector='b',alphavector=0.5)

    ax_reg, ax_rib,cs1 =plot_map_loadings(loadings_t.loc[3, :].filter(regex='T_'), None, None,
                      None, T_stalatlon, raster_lat,
                      raster_lon, raster_val, nb_pc=2, name='T',outpath=None, ax1=ax5, ribeirao=True,colorbaraltitude=True, colorvector='b',alphavector=0.5)


    ax1.set_title('$PC3_{UV}$ loadings')
    ax2.set_title('$PC4_{UV}$ loadings')
    ax3.set_title('$PC4_{TUV}$ loadings')
    ax4.set_title('$PC4_{QUV}$ loadings')
    ax5.set_title('$PC3_{T}$ loadings')


    for t, ax in zip(["a)","b)",'c)','d)','e)', 'f)', 'g)','h)','i)','j)'],[ax1, ax2,ax3, ax4, ax5]):
        ax.text(-0.05, 1.05, t, transform=ax.transAxes, size=14, weight='bold')

    # ax1.set_ylabel('PC2 loadings', labelpad=30)
    # ax3.set_ylabel('PC4 loadings', labelpad=30)
    # #
    # cs1.set_label('Temperature and wind', labelpad=-50)
    # cs2.set_label('Specific humidity and wind', labelpad=-50)


    plt.tight_layout()

    plt.savefig('/home/thomas/phd/framework/res/regional/chapIII/Fig6_ribeirao.png', dpi=300 )
    # plt.show()
    plt.close('all')

def plot_map_pos_stations_fig1():
    llcrnrlon = -49
    urcrnrlon = -43
    llcrnrlat = -26
    urcrnrlat = -21

    # df_u = pd.read_csv("/home/thomas/phd/framework/qa/out/qa_visual/Uw.csv", index_col=0, parse_dates=True).resample('H').mean()
    # df_v = pd.read_csv("/home/thomas/phd/framework/qa/out/qa_visual/Vw.csv", index_col=0, parse_dates=True).resample('H').mean()

    total_stations_used = list(set(staname_t+ staname_wind+ staname_q+ df_u.columns))
    AttSta = att_sta("/home/thomas/phd/obs/staClim/metadata/database_metadata.csv")
    AttSta.attributes = AttSta.attributes.drop(['peg','svg'])
    stalatlon = AttSta.attributes.loc[total_stations_used, ['Lat','Lon','network']]

    f, ax = matplotlib.pyplot.subplots()
    clevs = np.linspace(200, 1800, 17)

    llcrnrlon=-49
    urcrnrlon=-43
    llcrnrlat=-25
    urcrnrlat=-21
    plt, map, cs = map_domain(raster_lat, raster_lon, raster_val,llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, ax=ax, clevs=clevs)
    # cs.cmap.set_under('w')
    # cs.cmap.set_over('k')
    # cs.set_clim(500, 1500)


        # plt.legend(loc=1, numpoints=1, framealpha=0.4, fontsize=11)
    map.colorbar(cs, label='Altitude (m)')


    parallels = np.arange(llcrnrlat, -urcrnrlat,1)
    map.drawparallels(parallels,labels=[1,0,0,0],fontsize=11, linewidth=0.2)
    # draw meridians
    meridians = np.arange(llcrnrlon, urcrnrlon,1)
    map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=11, linewidth=0.2)

    map.drawmapscale(urcrnrlon-1.5, llcrnrlat+1.5, urcrnrlon, llcrnrlat, 50, barstyle='fancy',fontsize=11,units='km')
    map = add_stations_positions(map,stalatlon,s=7)


    # Put name on maps
    # x, y = map( (-23.036265 , -43.192749) # Rio
    plt.annotate('Rio De Janeiro', xy=map( -44.892749, -23.036265), fontsize=10)
    # plt.annotate('S\~{a}o Jose dos Campos', xy=map( -45.887146, -23.254973), fontsize=9)
    plt.annotate('Itajuba', xy=map( -45.431213, -22.429817), fontsize=10)
    plt.annotate('Volta redonda', xy=map( -44.079895, -22.541479), fontsize=10)

    plt.annotate('Campinas', xy=map( -47.05719, -22.946759), fontsize=10)
    # plt.annotate('Pouso Alegre', xy=map( -45.931091, -22.257072), fontsize=9)
    plt.annotate('S\~{a}o Carlos', xy=map( -47.892151,-22.023018), fontsize=10)
    plt.annotate('Varginas', xy=map( -45.431213, -21.574186), fontsize=10)
    plt.annotate('Santos', xy=map(-46.323853, -24.053487), fontsize=10)


    plt.annotate('Serra Da Mantiqueira', xy=map( -45.752063, -21.974662),color='k', fontsize=12, rotation=25)
    plt.annotate('Paraiba Valley', xy=map( -45.403748, -22.636642),color='k', fontsize=12, rotation=25)
    plt.annotate('Serra Do Mar', xy=map( -45.411987, -23.130751),color='k', fontsize=12, rotation=25)
    plt.annotate('Atlantic Ocean', xy=map( -46.181519 ,-24.51756),color='k', fontsize=12, rotation=25)
    plt.annotate('Mogi Valley', xy=map( -47.861938, -22.904768),color='k', fontsize=12, rotation=25)

    # map.plot(map(-43.392749, -23.036265),'o', markersize=30)
    # map.plot(-45.887146, -23.254973,'o', markersize=30)
    # map.plot( -45.431213, -22.429817,'o', markersize=30)
    # map.plot( -44.079895, -22.541479,'o', markersize=30)
    #
    # map.plot(-47.05719, -22.946759,'o', markersize=30)
    # map.plot(-45.931091, -22.257072,'o', markersize=30)
    # map.plot( -47.892151,-22.023018,'o', markersize=30)
    # map.plot( -45.431213, -21.574186,'o', markersize=30)
    # map.plot(-46.323853, -24.053487,'o', markersize=30)


    # map.plot(x, y, 'bo', markersize=18)
    # -21.574186 , -45.431213 # Varginas
    # -22.774662 , -45.552063 # Serra Da Mantiqueira
    # -22.936642 , -45.403748 # Valley do Paraiba
    # -22.904768 , -47.861938 # Mogi valley
    # (-23.530751 , -45.411987 # Serra Do Mar
    # -24.053487 , -46.323853 # Santos



    inpath ="/home/thomas/phd/geomap/data/shapefile/"

    # Cantareira
    shapefile= inpath+ 'cantareira/cantareiraWGS'
    map.readshapefile(shapefile, 'cantareiraWGS', drawbounds=True, linewidth=1.5, color='#48a3e6')

    # RMSP
    shapefile= inpath+ 'rmsp/rmsp_polig'
    map.readshapefile(shapefile, 'rmsp_polig', drawbounds=True, linewidth=1.5, color='#890045')
    plt.legend(loc=4, numpoints=1, framealpha=0.4, fontsize=11)

    plt.savefig('/home/thomas/phd/framework/res/regional/chapIII/Fig1_mapstations.pdf')
    plt.close('all')
    # plt.show()

def rec_T():
    pc1_t = dfs_pcs_reconstruct_t[1]
    pc2_t = dfs_pcs_reconstruct_t[2]
    pc3_t = dfs_pcs_reconstruct_t[3]

    pc1_t = pc1_t.multiply(df_t_w, axis=0)
    pc2_t = pc2_t .multiply(df_t_w, axis=0)
    pc3_t = pc3_t .multiply(df_t_w, axis=0)

    pctot_t = pc1_t + pc2_t + pc3_t
    pctot_t = pctot_t + df_t_mean

    pc1_t = pc1_t.groupby(lambda x:x.hour).mean().filter(regex='T_')
    pc2_t = pc2_t.groupby(lambda x:x.hour).mean().filter(regex='T_')
    pc3_t = pc3_t.groupby(lambda x:x.hour).mean().filter(regex='T_')
    pctot_t = pctot_t.groupby(lambda x:x.hour).mean().filter(regex='T_')


    daily_scores =scores_t.groupby(lambda t: (t.hour)).mean()
    daily_scores.columns = ['1','2', '3']

    for h in range(24):
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
        f.set_figheight(10)
        f.set_figwidth(15)
        plt.subplots_adjust(wspace=0.05,hspace=0.05)

        ax_reg1, ax_rib,cs1 = plot_map_loadings(pc1_t.iloc[h,:].filter(regex='T_'), None, None,
                              None, T_stalatlon, raster_lat,
                              raster_lon, raster_val, nb_pc=1, name='T',outpath=None, ax1=ax1, ribeirao=False, vmin=pc1_t.min().min(), vmax=pc1_t.max().max())

        ax_reg2, ax_rib,cs2 = plot_map_loadings(pc2_t.iloc[h,:].filter(regex='T_'), None, None,
                              None, T_stalatlon, raster_lat,
                              raster_lon, raster_val, nb_pc=2, name='T',outpath=None, ax1=ax2, ribeirao=False, vmin=pc2_t.min().min(), vmax=pc2_t.max().max())

        ax_reg3, ax_rib,cs1 = plot_map_loadings(pc3_t.iloc[h,:].filter(regex='T_'), None, None,
                      None, T_stalatlon, raster_lat,
                      raster_lon, raster_val, nb_pc=3, name='T',outpath=None, ax1=ax3, ribeirao=False, vmin=pc3_t.min().min(), vmax=pc3_t.max().max())

        ax_reg4, ax_rib,cs1 = plot_map_loadings(pctot_t.iloc[h,:].filter(regex='T_'), None, None,
                      None, T_stalatlon, raster_lat,
                      raster_lon, raster_val, nb_pc=3, name='T',outpath=None, ax1=ax4, ribeirao=False, vmin=pctot_t.min().min(), vmax=pctot_t.max().max())


        ax1.set_title(r'PC1 ($63\%$)')
        ax2.set_title(r'PC2 ($9\%$)')
        ax3.set_title(r'PC3 ($6\%$)')
        ax4.set_title(r'Total ($79\%$)')


        daily_scores.plot(ax=ax6, color=['b','r','k'], linewidth=3, legend=None)
        ax6.legend(bbox_to_anchor=(-0.2, 1), loc=0, borderaxespad=0.)

        ax6.scatter([h]*3,daily_scores.iloc[h],s=80,c='k', alpha=0.5)
        ax6.axvline(x=h,c='0.5', alpha=0.5)
        ax6.axhline(0,c='0.5', alpha=0.5)
        ax6.set_xlabel('Hours')
        ax6.set_ylabel('PC scores')

        f.delaxes(ax5)
        plt.suptitle("Time: " + str(str(h).zfill(2))+"H LT", fontsize=20)
        # plt.tight_layout()
        outfilename = "/home/thomas/phd/framework/res/regional/rec_anim/T/" + str(h).zfill(2)+"rec_T.png"
        print outfilename
        plt.savefig(outfilename , bbox_inches='tight', dpi=300 )
        # return map
        plt.close('all')

def rec_Q():
    pc1_q = dfs_pcs_reconstruct_q[1]
    pc2_q = dfs_pcs_reconstruct_q[2]
    pc3_q = dfs_pcs_reconstruct_q[3]

    pc1_q = pc1_q.multiply(df_q_w, axis=0)
    pc2_q = pc2_q .multiply(df_q_w, axis=0)
    pc3_q = pc3_q .multiply(df_q_w, axis=0)

    pctot_q = pc1_q + pc2_q + pc3_q
    pctot_q = pctot_q + df_q_mean

    pc1_q = pc1_q.groupby(lambda x:x.hour).mean().filter(regex='Q_')
    pc2_q = pc2_q.groupby(lambda x:x.hour).mean().filter(regex='Q_')
    pc3_q = pc3_q.groupby(lambda x:x.hour).mean().filter(regex='Q_')
    pctot_q = pctot_q.groupby(lambda x:x.hour).mean().filter(regex='Q_')


    daily_scores =scores_q.groupby(lambda t: (t.hour)).mean()
    daily_scores.columns = ['1','2', '3']

    for h in range(24):
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
        f.set_figheight(10)
        f.set_figwidth(15)
        plt.subplots_adjust(wspace=0.05,hspace=0.05)

        ax_reg1, ax_rib,cs1 = plot_map_loadings(pc1_q.iloc[h,:].filter(regex='Q_'), None, None,
                              None, Q_stalatlon, raster_lat,
                              raster_lon, raster_val, nb_pc=1, name='Q',outpath=None, ax1=ax1, ribeirao=False, vmin=pc1_q.min().min(), vmax=pc1_q.max().max())

        ax_reg2, ax_rib,cs2 = plot_map_loadings(pc2_q.iloc[h,:].filter(regex='Q_'), None, None,
                              None, Q_stalatlon, raster_lat,
                              raster_lon, raster_val, nb_pc=2, name='Q',outpath=None, ax1=ax2, ribeirao=False, vmin=pc2_q.min().min(), vmax=pc2_q.max().max())

        ax_reg3, ax_rib,cs1 = plot_map_loadings(pc3_q.iloc[h,:].filter(regex='Q_'), None, None,
                      None, Q_stalatlon, raster_lat,
                      raster_lon, raster_val, nb_pc=3, name='Q',outpath=None, ax1=ax3, ribeirao=False, vmin=pc3_q.min().min(), vmax=pc3_q.max().max())

        ax_reg4, ax_rib,cs1 = plot_map_loadings(pctot_q.iloc[h,:].filter(regex='Q_'), None, None,
                      None, Q_stalatlon, raster_lat,
                      raster_lon, raster_val, nb_pc=3, name='Q',outpath=None, ax1=ax4, ribeirao=False, vmin=pctot_q.min().min(), vmax=pctot_q.max().max())


        ax1.set_title(r'PC1 ($72\%$)')
        ax2.set_title(r'PC2 ($6\%$)')
        ax3.set_title(r'PC3 ($5\%$)')
        ax4.set_title(r'Total ($83\%$)')


        daily_scores.plot(ax=ax6, color=['b','r','k'], linewidth=3, legend=None)
        ax6.legend(bbox_to_anchor=(-0.2, 1), loc=0, borderaxespad=0.)

        ax6.scatter([h]*3,daily_scores.iloc[h],s=80,c=['b','r','k'], alpha=0.5)
        ax6.axvline(x=h,c='k')
        ax6.axhline(0,c='0.5', alpha=0.5)
        ax6.set_xlabel('Hours')
        ax6.set_ylabel('PC scores')

        f.delaxes(ax5)
        plt.suptitle("Time: " + str(str(h).zfill(2))+"H LT", fontsize=20)
        # plt.tight_layout()
        outfilename = "/home/thomas/phd/framework/res/regional/rec_anim/Q/" + str(h).zfill(2)+"rec_q.png"
        print outfilename
        plt.savefig(outfilename , bbox_inches='tight', dpi=300 )
        # return map
        plt.close('all')

def rec_UV():

    pc1_uv = dfs_pcs_reconstruct_uv[1]
    pc2_uv = dfs_pcs_reconstruct_uv[2]
    pc3_uv = dfs_pcs_reconstruct_uv[3]
    pc4_uv = dfs_pcs_reconstruct_uv[4]

    pc1_u = pc1_uv.filter(regex='U_')
    pc2_u = pc2_uv.filter(regex='U_')
    pc3_u = pc3_uv.filter(regex='U_')
    pc4_u = pc4_uv.filter(regex='U_')

    pc1_v = pc1_uv.filter(regex='V_')
    pc2_v = pc2_uv.filter(regex='V_')
    pc3_v = pc3_uv.filter(regex='V_')
    pc4_v = pc4_uv.filter(regex='V_')

    pc1_u = pc1_u.multiply(df_u_w, axis=0)
    pc2_u = pc2_u.multiply(df_u_w, axis=0)
    pc3_u = pc3_u.multiply(df_u_w, axis=0)
    pc4_u = pc4_u.multiply(df_u_w, axis=0)

    pc1_v = pc1_v.multiply(df_v_w, axis=0)
    pc2_v = pc2_v.multiply(df_v_w, axis=0)
    pc3_v = pc3_v.multiply(df_v_w, axis=0)
    pc4_v = pc4_v.multiply(df_v_w, axis=0)

    pctot_u = pc1_u + pc2_u + pc3_u + pc4_u
    pctot_u = pctot_u + df_u_mean

    pctot_v = pc1_v + pc2_v + pc3_v + pc4_v
    pctot_v = pctot_v + df_v_mean

    pc1_u = pc1_u.groupby(lambda x:x.hour).mean()
    pc2_u = pc2_u.groupby(lambda x:x.hour).mean()
    pc3_u = pc3_u.groupby(lambda x:x.hour).mean()
    pc4_u = pc4_u.groupby(lambda x:x.hour).mean()
    pctot_u = pctot_u.groupby(lambda x:x.hour).mean()

    pc1_v = pc1_v.groupby(lambda x:x.hour).mean()
    pc2_v = pc2_v.groupby(lambda x:x.hour).mean()
    pc3_v = pc3_v.groupby(lambda x:x.hour).mean()
    pc4_v = pc4_v.groupby(lambda x:x.hour).mean()
    pctot_v = pctot_v.groupby(lambda x:x.hour).mean()

    pc1_uv = pd.concat([pc1_u, pc1_v], axis=1, join='inner')
    pc2_uv = pd.concat([pc2_u, pc2_v], axis=1, join='inner')
    pc3_uv = pd.concat([pc3_u, pc3_v], axis=1, join='inner')
    pc4_uv = pd.concat([pc4_u, pc4_v], axis=1, join='inner')
    pctot_uv = pd.concat([pctot_u, pctot_v], axis=1, join='inner')

    daily_scores =scores_uv.groupby(lambda t: (t.hour)).mean()
    daily_scores.columns = ['1','2', '3','4']

    # 2008 - 2015 mean observations

    df_u = pd.read_csv("/home/thomas/phd/framework/qa/out/qa_visual/Uw.csv", index_col=0, parse_dates=True).resample('H').mean()
    df_v = pd.read_csv("/home/thomas/phd/framework/qa/out/qa_visual/Vw.csv", index_col=0, parse_dates=True).resample('H').mean()
    # df_u = apply_func(df_u, normalize)
    # df_v = apply_func(df_v, normalize)
    stanames_obs = df_u.columns
    # Drop ribeirao stations
    stanames_obs  = stanames_obs.drop(['C08', 'C07','C06', 'C05','C04', 'C10','C11', 'C12','C13', 'C14','C15', 'C16', 'C17','C18','C19'])
    df_u_obs = df_u.loc[:, stanames_obs]
    df_v_obs = df_v.loc[:, stanames_obs]
    # group
    df_u_daily_obs = df_u_obs.groupby(lambda x: x.hour).mean()
    df_v_daily_obs = df_v_obs.groupby(lambda x: x.hour).mean()
    AttSta = att_sta("/home/thomas/phd/obs/staClim/metadata/database_metadata.csv")
    print AttSta.attributes
    stalatlon_wind_obs = AttSta.attributes.loc[stanames_obs, ['Lat','Lon','network']]

    for h in range(24):
        # f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
        # f.set_figheight(10)
        # f.set_figwidth(15)
        # plt.subplots_adjust(wspace=0.05,hspace=0.05)
        #
        # ax_reg, ax_rib,cs1 =plot_map_loadings(None, pc1_uv.iloc[h, :].filter(regex='U_'), pc1_uv.iloc[h, :].filter(regex='V_'),
        #                   wind_stalatlon, None, raster_lat,
        #                   raster_lon, raster_val, nb_pc=1, name='UV',outpath=None, ax1=ax1, ax3 = None,
        #                                       ribeirao=False,colorvector='b', vmin=pc1_uv.min().min(), vmax=pc1_uv.max().max(),linewidth=0.01, width=0.01,
        #                                       scale=4, scale_quiverkey=0.5, quiver_legend=r'$0.5 m.s^{-1}$')
        #
        # ax_reg, ax_rib,cs2 =plot_map_loadings(None, pc2_uv.iloc[h, :].filter(regex='U_'), pc2_uv.iloc[h, :].filter(regex='V_'),
        #                   wind_stalatlon, None, raster_lat,
        #                   raster_lon, raster_val, nb_pc=2, name='UV',outpath=None, ax1=ax2, ax3 = None,
        #                                       ribeirao=False,colorvector='b', vmin=pc2_uv.min().min(), vmax=pc2_uv.max().max(),linewidth=0.01, width=0.01,
        #                                       scale=4, scale_quiverkey=0.5, quiver_legend=r'$0.5 m.s^{-1}$')
        #
        # ax_reg, ax_rib,cs3 =plot_map_loadings(None, pc3_uv.iloc[h, :].filter(regex='U_'), pc3_uv.iloc[h, :].filter(regex='V_'),
        #                   wind_stalatlon, None, raster_lat,
        #                   raster_lon, raster_val, nb_pc=3, name='UV',outpath=None, ax1=ax3, ax3 = None,
        #                                       ribeirao=False,colorvector='b', vmin=pc3_uv.min().min(), vmax=pc3_uv.max().max(),linewidth=0.01, width=0.01,
        #                                       scale=4, scale_quiverkey=0.5, quiver_legend=r'$0.5 m.s^{-1}$')
        #
        # ax_reg, ax_rib,cs4 =plot_map_loadings(None, pc4_uv.iloc[h, :].filter(regex='U_'), pc4_uv.iloc[h, :].filter(regex='V_'),
        #                   wind_stalatlon, None, raster_lat,
        #                   raster_lon, raster_val, nb_pc=4, name='UV',outpath=None, ax1=ax4, ax3 = None,
        #                                       ribeirao=False,colorvector='b', vmin=pc4_uv.min().min(), vmax=pc4_uv.max().max(),linewidth=0.01, width=0.01,
        #                                       scale=4, scale_quiverkey=0.5, quiver_legend=r'$0.5 m.s^{-1}$')
        #
        #
        # ax1.set_title(r'PC1 ($25\%$)')
        # ax2.set_title(r'PC2 ($13\%$)')
        # ax3.set_title(r'PC3 ($6\%$)')
        # ax4.set_title(r'PC4 ($4\%$)')
        #
        # daily_scores.plot(ax=ax6, color=['b','r','k','g'], linewidth=3, legend=None)
        # ax6.legend(bbox_to_anchor=(-0.2, 1), loc=0, borderaxespad=0.)
        #
        # ax6.scatter([h]*4,daily_scores.iloc[h],s=80,c=['b','r','k','g'], alpha=0.5)
        # ax6.axvline(x=h,c='k')
        # ax6.axhline(0,c='0.5', alpha=0.5)
        # ax6.set_xlabel('Hours')
        # ax6.set_ylabel('PC scores')
        #
        # f.delaxes(ax5)
        # plt.suptitle("Time: " + str(str(h).zfill(2))+"H LT", fontsize=20)
        # # plt.tight_layout()
        # outfilename = "/home/thomas/phd/framework/res/regional/rec_anim/UV/" + str(h).zfill(2)+"rec_UV.png"
        # print outfilename
        # plt.savefig(outfilename , bbox_inches='tight', dpi=300 )
        # # return map
        # plt.close('all')

        f, (ax1, ax2) = plt.subplots(1,2)
        f.set_figheight(5)
        f.set_figwidth(10)
        # plt.subplots_adjust(wspace=0.05,hspace=0.05)

        ax_reg, ax_rib,cs5 =plot_map_loadings(None, pctot_uv.iloc[h, :].filter(regex='U_'), pctot_uv.iloc[h, :].filter(regex='V_'),
              wind_stalatlon, None, raster_lat,
              raster_lon, raster_val, nb_pc=4, name='UV',outpath=None, ax1=ax1, ax3 = None,ykey=1.15,
                                  ribeirao=False,colorvector='b', vmin=pctot_uv.min().min(), vmax=pctot_uv.max().max(),scale=15, scale_quiverkey=2, quiver_legend=r'$2 m.s^{-1}$',  linewidth=0.0015, width=0.003)


        ax_reg, ax_rib,cs6 =plot_map_loadings(None, df_u_daily_obs.iloc[h, :], df_v_daily_obs.iloc[h, :],
              stalatlon_wind_obs, None, raster_lat,
              raster_lon, raster_val, nb_pc=4, name='UV',outpath=None, ax1=ax2, ax3 = None,ykey=1.15,
                                  ribeirao=False,colorvector='b',scale=15, scale_quiverkey=2, quiver_legend=r'$2 m.s^{-1}$',  linewidth=0.001, width=0.0025)

        ax1.set_title(r'PCA reconstructed wind ($48\%$)')
        ax2.set_title(r'Mean observed wind (2008-2015)')


        plt.suptitle("Time: " + str(str(h).zfill(2))+"H LT", fontsize=20)
        plt.tight_layout()
        outfilename = "/home/thomas/phd/framework/res/regional/rec_anim/UV/tot/" + str(h).zfill(2)+"rec_UV.png"
        print outfilename
        plt.savefig(outfilename , bbox_inches='tight', dpi=300 )
        # return map
        plt.close('all')


def rec_TUV_QUV():
    pc2_tuv = dfs_pcs_reconstruct_tuv[2]
    pc4_tuv = dfs_pcs_reconstruct_tuv[4]

    pc2_quv = dfs_pcs_reconstruct_quv[2]
    pc4_quv = dfs_pcs_reconstruct_quv[4]

    # TUV
    pc2_tuv_u = pc2_tuv.filter(regex='U_')
    pc4_tuv_u = pc4_tuv.filter(regex='U_')
    pc2_tuv_v = pc2_tuv.filter(regex='V_')
    pc4_tuv_v = pc4_tuv.filter(regex='V_')
    pc2_tuv_t = pc2_tuv.filter(regex='T_')
    pc4_tuv_t = pc4_tuv.filter(regex='T_')

    # QUV
    pc2_quv_u = pc2_quv.filter(regex='U_')
    pc4_quv_u = pc4_quv.filter(regex='U_')
    pc2_quv_v = pc2_quv.filter(regex='V_')
    pc4_quv_v = pc4_quv.filter(regex='V_')
    pc2_quv_q = pc2_quv.filter(regex='Q_')
    pc4_quv_q = pc4_quv.filter(regex='Q_')

    # Rescale TUV
    pc2_tuv_u = pc2_tuv_u.multiply(df_u_w, axis=0)
    pc4_tuv_u = pc4_tuv_u.multiply(df_u_w, axis=0)
    pc2_tuv_v = pc2_tuv_v.multiply(df_v_w, axis=0)
    pc4_tuv_v = pc4_tuv_v.multiply(df_v_w, axis=0)
    pc2_tuv_t = pc2_tuv_t.multiply(df_t_w, axis=0)
    pc4_tuv_t = pc4_tuv_t.multiply(df_t_w, axis=0)

    # Rescale QUV
    pc2_quv_u = pc2_quv_u.multiply(df_u_w, axis=0)
    pc4_quv_u = pc4_quv_u.multiply(df_u_w, axis=0)
    pc2_quv_v = pc2_quv_v.multiply(df_v_w, axis=0)
    pc4_quv_v = pc4_quv_v.multiply(df_v_w, axis=0)
    pc2_quv_q = pc2_quv_q.multiply(df_q_w, axis=0)
    pc4_quv_q = pc4_quv_q.multiply(df_q_w, axis=0)


    # Daily average
    pc2_tuv_u = pc2_tuv_u.groupby(lambda x:x.hour).mean()
    pc4_tuv_u = pc4_tuv_u.groupby(lambda x:x.hour).mean()
    pc2_tuv_v = pc2_tuv_v.groupby(lambda x:x.hour).mean()
    pc4_tuv_v = pc4_tuv_v.groupby(lambda x:x.hour).mean()
    pc2_tuv_t = pc2_tuv_t.groupby(lambda x:x.hour).mean()
    pc4_tuv_t = pc4_tuv_t.groupby(lambda x:x.hour).mean()

    pc2_quv_u = pc2_quv_u.groupby(lambda x:x.hour).mean()
    pc4_quv_u = pc4_quv_u.groupby(lambda x:x.hour).mean()
    pc2_quv_v = pc2_quv_v.groupby(lambda x:x.hour).mean()
    pc4_quv_v = pc4_quv_v.groupby(lambda x:x.hour).mean()
    pc2_quv_q = pc2_quv_q.groupby(lambda x:x.hour).mean()
    pc4_quv_q = pc4_quv_q.groupby(lambda x:x.hour).mean()



    # Recreate dataframe
    pc2_tuv = pd.concat([pc2_tuv_u, pc2_tuv_v, pc2_tuv_t], axis=1, join='inner')
    pc4_tuv = pd.concat([pc4_tuv_u, pc4_tuv_v, pc4_tuv_t], axis=1, join='inner')

    pc2_quv = pd.concat([pc2_quv_u, pc2_quv_v, pc2_quv_q], axis=1, join='inner')
    pc4_quv = pd.concat([pc4_quv_u, pc4_quv_v, pc4_quv_q], axis=1, join='inner')

    daily_scores_tuv =scores_tuv.loc[:,[2]].groupby(lambda t: (t.hour)).mean()
    daily_scores_quv =scores_quv.loc[:,[2]].groupby(lambda t: (t.hour)).mean()


    for h in range(24):
        f, ((ax1, ax2), (ax5, ax6)) = plt.subplots(2,2)
        f.set_figheight(15)
        f.set_figwidth(15)
        plt.subplots_adjust(wspace=0.2,hspace=0.2)

        ax_reg, ax_rib,cs1 =plot_map_loadings(pc2_tuv.iloc[h, :].filter(regex='T_'), pc2_tuv.iloc[h, :].filter(regex='U_'), pc2_tuv.iloc[h, :].filter(regex='V_'),
                          wind_stalatlon, T_stalatlon, raster_lat,
                          raster_lon, raster_val, nb_pc=2, name='UV',outpath=None, ax1=ax1, ribeirao=False, alphavector=0.5, colorvector='b', ykey=1.25,
                                              vmin=pc2_tuv_t.min().min(), vmax=pc2_tuv_t.max().max(),scale=5, scale_quiverkey=0.5, quiver_legend=r'$0.5 m.s^{-1}$')

        cs1.set_label(r'Temperature ($^{\circ}C$)')


        #
        # ax_reg, ax_rib,cs2 =plot_map_loadings(pc4_tuv.iloc[h, :].filter(regex='T_'), pc4_tuv.iloc[h, :].filter(regex='U_'), pc4_tuv.iloc[h, :].filter(regex='V_'),
        #               wind_stalatlon, T_stalatlon, raster_lat,
        #               raster_lon, raster_val, nb_pc=2, name='UV',outpath=None, ax1=ax3, ribeirao=True, alphavector=0.5, colorvector='b', ykey=1.25,
        #                                       vmin=pc4_tuv_t.min().min(), vmax=pc4_tuv_t.max().max(),scale=5, scale_quiverkey=0.5, quiver_legend=r'$0.5 m.s^{-1}$')


        ax_reg, ax_rib,cs3 =plot_map_loadings(pc2_quv.iloc[h, :].filter(regex='Q_'), pc2_quv.iloc[h, :].filter(regex='U_'), pc2_quv.iloc[h, :].filter(regex='V_'),
                          wind_stalatlon, Q_stalatlon, raster_lat,
                          raster_lon, raster_val, nb_pc=4, name='UV',outpath=None, ax1=ax2, ribeirao=False, alphavector=0.5, colorvector='b', ykey=1.25,
                                              vmin=pc2_quv_q.min().min(), vmax=pc2_quv_q.max().max(),scale=5, scale_quiverkey=0.5, quiver_legend=r'$0.5 m.s^{-1}$')
        cs3.set_label(r'Specific humidity ($g.kg^{-1}$)')

        # ax_reg, ax_rib,cs4 =plot_map_loadings(pc4_quv.iloc[h, :].filter(regex='Q_'), pc4_quv.iloc[h, :].filter(regex='U_'), pc4_quv.iloc[h, :].filter(regex='V_'),
        #               wind_stalatlon, Q_stalatlon, raster_lat,
        #               raster_lon, raster_val, nb_pc=4, name='UV',outpath=None, ax1=ax4, ribeirao=True, alphavector=0.5, colorvector='b', ykey=1.25,
        #                                       vmin=pc4_quv_q.min().min(), vmax=pc4_quv_q.max().max(),scale=5, scale_quiverkey=0.5, quiver_legend=r'$0.5 m.s^{-1}$')
        #
        # ax_reg, ax_rib,cs3 = plot_map_loadings(pc4_quv.iloc[h, :].filter(regex='Q_'), None, None,
        #                       None, Q_stalatlon, raster_lat,
        #                       raster_lon, raster_val, nb_pc=3, name='Q',outpath=None, ax1=ax8, ax3 = None, ribeirao=False, vmin=pc4_quv_q.min().min(), vmax=pc4_quv_q.max().max())


        daily_scores_tuv.plot(ax=ax5, color='r', linewidth=3, legend=None)
        # ax5.legend(bbox_to_anchor=(-0.2, 1), loc=0, borderaxespad=0.)
        ax5.scatter([h],daily_scores_tuv.iloc[h],s=80,c='k', alpha=0.5)
        ax5.axvline(x=h,c='k')
        ax5.axhline(0,c='k')
        ax5.set_xlabel('Hours')
        ax5.set_ylabel('PC scores')

        daily_scores_quv.plot(ax=ax6, color='r', linewidth=3, legend=None)
        # ax6.legend(bbox_to_anchor=(-0.2, 1), loc=0, borderaxespad=0.)
        ax6.scatter([h],daily_scores_quv.iloc[h],s=80,c='k', alpha=0.5)
        ax6.axvline(x=h,c='k')
        ax6.axhline(0,c='k')
        ax6.set_xlabel('Hours')
        # ax6.set_ylabel('PC scores')

        ax1.set_title(r'PC2 TUV ($15\%$)')
        ax2.set_title(r'PC2 QUV ($15\%$)')
        # ax3.set_title('PC4 TUV')
        # ax4.set_title('PC4 QUV')
        # ax8.set_title('PC4 QUV')

        ax5.set_title('TUV')
        ax6.set_title('QUV')


        # f.delaxes(ax7)
        plt.suptitle("Time: " + str(str(h).zfill(2))+"H LT", fontsize=24)
        # plt.tight_layout()
        outfilename = "/home/thomas/phd/framework/res/regional/rec_anim/TUV_QUV/" + str(h).zfill(2)+"rec_TUV_QUV.png"
        print outfilename
        plt.savefig(outfilename , bbox_inches='tight', dpi=300 )
        # return map
        plt.close('all')


if __name__ == "__main__":

    # df_t, df_q,df_u, df_v, df_tuv,df_quv,  df_tquv, df_uv, staname_t, staname_q, staname_wind,\
    # df_t_mean, df_q_mean,df_u_mean, df_v_mean, df_t_w, df_q_w,df_u_w, df_v_w, df_t_old, df_q_old,df_u_old, df_v_old = get_dataframes()
    # raster_val, raster_lon, raster_lat, wind_stalatlon, T_stalatlon, Q_stalatlon = get_background_and_pos(staname_t, staname_q, staname_wind)
    #
    # outpath ="../../res/regional/combined_norm/"
    #

    # #=============================================================================
    # # Perform PCA
    # #=============================================================================
    # # nb_pc = 7
    # stamod_t, loadings_t, scores_t ,dfs_pcs_reconstruct_t= do_regional_pca(df_t, 3, outpath = outpath + "T/")
    # stamod_q, loadings_q, scores_q ,dfs_pcs_reconstruct_q= do_regional_pca(df_q, 3, outpath = outpath + "Q/")
    # # stamod_u, loadings_u, scores_u ,dfs_pcs_reconstruct_u, ax_daily_T = do_regional_pca(df_u, nb_pc, outpath = outpath + "U/")
    # # stamod_v, loadings_v, scores_v ,dfs_pcs_reconstruct_v, ax_daily_T = do_regional_pca(df_v, nb_pc, outpath = outpath + "V/")
    # stamod_tuv, loadings_tuv, scores_tuv ,dfs_pcs_reconstruct_tuv = do_regional_pca(df_tuv, 4, outpath = outpath + "TUV/")
    # stamod_quv, loadings_quv, scores_quv ,dfs_pcs_reconstruct_quv= do_regional_pca(df_quv, 4, outpath = outpath + "QUV/")
    # stamod_uv, loadings_uv, scores_uv ,dfs_pcs_reconstruct_uv= do_regional_pca(df_uv, 4, outpath = outpath + "UV/")
    # # stamod_tquv, loadings_tquv, scores_tquv ,dfs_pcs_reconstruct_tquv = do_regional_pca(df_tquv, nb_pc, outpath = outpath + "TQUV/")

    #=============================================================================
    # Figures article
    #=============================================================================
    print('Plot map stations')
    plot_map_pos_stations_fig1()
    # print('Plot temperature')
    # plot_fig2_T()
    # print('Plot humiodity')
    # plot_fig3_Q()
    # print('plot wind')
    # plot_fig4_UV()
    # print('plot tuv quv')
    # plot_fig5_TUV_QUV()
    # print('Ribeiroa')
    # plot_fig6_ribeirao()

    #=============================================================================
    # Figures presentation reconstruction
    #=============================================================================

    # rec_Q()
    # rec_T()
    # rec_UV()
    # rec_TUV_QUV()


# ################################################################################################################################
# #   Old grafics
# #################################################################################################################################
#     #=============================================================================
#     # Save the results
#     #=============================================================================
#     # loadings.to_csv('../out/regional/loadings_ctquv.csv')
#     # scores.to_csv('../out/regional/scores_ctquv.csv')
#
#     #=============================================================================
#     # Plot Daily reconstructed PCs and loadings
#     #=============================================================================
#     #
#     # for npc in range(1, nb_pc+1):
#     #     #
#     #     # Temperature
#     #     print('Temperature')
#     #     rec_daily = dfs_pcs_reconstruct_t[npc].groupby(lambda t: (t.hour)).mean()
#     #     load_t = loadings_t.loc[npc, :]
#     #
#     #     # plot_daily_map_pc(rec_daily.filter(regex='T_'), None, None,
#     #     #                   None, T_stalatlon, raster_lat, raster_lon, raster_val,
#     #     #             nb_pc=npc, name='T',outpath=outpath+ "T/reconstruct")
#     #
#     #     plt.close('all')
#     #     plot_map_loadings(load_t.filter(regex='T_'), None, None,
#     #                       None, T_stalatlon, raster_lat,
#     #                       raster_lon, raster_val, nb_pc=npc, name='T',outpath=outpath+ "T/loadings/")
#     #
#     #     # Humidity
#     #     print('Humidity')
#     #     rec_daily = dfs_pcs_reconstruct_q[npc].groupby(lambda t: (t.hour)).mean()
#     #     load_q = loadings_q.loc[npc, :]
#     #
#     #     # plot_daily_map_pc(rec_daily.filter(regex='Q_'), None, None,
#     #     #                   None, Q_stalatlon, raster_lat, raster_lon, raster_val,
#     #     #             nb_pc=npc, name='Q',outpath=outpath+ "Q/reconstruct")
#     #
#     #     plot_map_loadings(load_q.filter(regex='Q_'), None, None,
#     #                       None, Q_stalatlon, raster_lat,
#     #                       raster_lon, raster_val, nb_pc=npc, name='Q',outpath=outpath+ "Q/loadings/")
#     #
#     #     # # U
#     #     # print('Zonal wind')
#     #     # rec_daily = dfs_pcs_reconstruct_u[npc].groupby(lambda t: (t.hour)).mean()
#     #     # load_u = loadings_u.loc[npc, :]
#     #     #
#     #     # # plot_daily_map_pc(None,rec_daily.filter(regex='U_'), np.zeros(load_u.filter(regex='U_').shape),
#     #     # #                   wind_stalatlon, None, raster_lat, raster_lon, raster_val,
#     #     # #             nb_pc=npc, name='U',outpath=outpath+ "U/reconstruct")
#     #     #
#     #     # plot_map_loadings(None,load_u.filter(regex='U_'), np.zeros(load_u.filter(regex='U_').shape),
#     #     #                   wind_stalatlon, None, raster_lat,
#     #     #                   raster_lon, raster_val, nb_pc=npc, name='U',outpath=outpath+ "U/loadings/")
#     #     #
#     #     # # V
#     #     # print('Meridional wind')
#     #     # rec_daily = dfs_pcs_reconstruct_v[npc].groupby(lambda t: (t.hour)).mean()
#     #     # load_v = loadings_v.loc[npc, :]
#     #     #
#     #     # # plot_daily_map_pc(None, np.zeros(load_v.filter(regex='V_').shape),rec_daily.filter(regex='V_'),
#     #     # #                   wind_stalatlon, None, raster_lat, raster_lon, raster_val,
#     #     # #             nb_pc=npc, name='V',outpath=outpath+ "V/reconstruct")
#     #     #
#     #     # plot_map_loadings(None, np.zeros(load_v.filter(regex='V_').shape),load_v.filter(regex='V_'),
#     #     #                   wind_stalatlon, None, raster_lat,
#     #     #                   raster_lon, raster_val, nb_pc=npc, name='V',outpath=outpath+ "V/loadings/")
#     #
#     #     # UV
#     #     print('UV')
#     #     rec_daily = dfs_pcs_reconstruct_uv[npc].groupby(lambda t: (t.hour)).mean()
#     #     load_uv = loadings_uv.loc[npc, :]
#     #
#     #     # plot_daily_map_pc(None,rec_daily.filter(regex='U_'), rec_daily.filter(regex='V_'),
#     #     #                   wind_stalatlon, None, raster_lat, raster_lon, raster_val,
#     #     #             nb_pc=npc, name='UV',outpath=outpath + "UV/reconstruct")
#     #
#     #     plot_map_loadings(None, load_uv.filter(regex='U_'), load_uv.filter(regex='V_'),
#     #                       wind_stalatlon, None, raster_lat,
#     #                       raster_lon, raster_val, nb_pc=npc, name='UV',outpath=outpath+"UV/loadings/")
#     #
#     #     # TUV
#     #     print('TUV')
#     #     rec_daily = dfs_pcs_reconstruct_tuv[npc].groupby(lambda t: (t.hour)).mean()
#     #     load_tuv = loadings_tuv.loc[npc, :]
#     #
#         # plot_daily_map_pc(rec_daily.filter(regex='T_'),rec_daily.filter(regex='U_'), rec_daily.filter(regex='V_'),
#         #                   wind_stalatlon, T_stalatlon, raster_lat, raster_lon, raster_val,
#         #             nb_pc=npc, name='TUV',outpath=outpath + "TUV/reconstruct/")
#
#         # plot_map_loadings(load_tuv.filter(regex='T_'), load_tuv.filter(regex='U_'), load_tuv.filter(regex='V_'),
#         #                   wind_stalatlon, T_stalatlon, raster_lat,
#         #                   raster_lon, raster_val, nb_pc=npc, name='TUV',outpath=outpath+"TUV/loadings/")
#     #
#     #     # QUV
#     #     print('QUV')
#     #     rec_daily = dfs_pcs_reconstruct_quv[npc].groupby(lambda t: (t.hour)).mean()
#     #     load_quv = loadings_quv.loc[npc, :]
#     #     #
#     #     # plot_daily_map_pc(rec_daily.filter(regex='Q_'),rec_daily.filter(regex='U_'), rec_daily.filter(regex='V_'),
#     #     #                   wind_stalatlon, Q_stalatlon, raster_lat, raster_lon, raster_val,
#     #     #             nb_pc=npc, name='QUV',outpath=outpath + "QUV/reconstruct")
#     #
#     #     plot_map_loadings(load_quv.filter(regex='Q_'), load_quv.filter(regex='U_'), load_quv.filter(regex='V_'),
#     #                       wind_stalatlon, Q_stalatlon, raster_lat,
#     #                       raster_lon, raster_val, nb_pc=npc, name='QUV',outpath=outpath+"QUV/loadings/")
#     #     #
#     #     #
#     #     # # # TQUV
#     #     # # print('TQUV')
#     #     # # rec_daily = dfs_pcs_reconstruct_tquv[npc].groupby(lambda t: (t.hour)).mean()
#     #     # # load_tquv = loadings_tquv.loc[npc, :]
#     #     # #
#     #     # # # plot_daily_map_pc(rec_daily.filter(regex='T_'),rec_daily.filter(regex='U_'), rec_daily.filter(regex='V_'),
#     #     # # #                   wind_stalatlon, T_stalatlon, raster_lat, raster_lon, raster_val,
#     #     # # #             nb_pc=npc, name='TQUV',outpath=outpath + "TQUV/reconstruct")
#     #     # #
#     #     # # plot_map_loadings(load_tquv.filter(regex='T_'), load_tquv.filter(regex='U_'), load_tquv.filter(regex='V_'),
#     #     # #                   wind_stalatlon, T_stalatlon, raster_lat,
#     #     # #                   raster_lon, raster_val, nb_pc=npc, name='TQUV',outpath=outpath+"TQUV/loadings/",use_pcvar2=True, pc_var2=load_tquv.filter(regex='Q_'), stalatlon2 = Q_stalatlon
#     #     # #                   )
