from mailib.decomposition.pca_regional_articleIII import *




if __name__ == "__main__":

    df_t, df_q,df_u, df_v, df_tuv, df_quv, df_tquv, df_uv, staname_t, staname_q, staname_wind = get_dataframes()
    # df = pd.concat([df_u,df_v], join='inner', axis=1)

    df = df_tuv

    df_6h = df.shift(6)
    df_12h = df.shift(12)
    df_18h = df.shift(18)

    df_0h = df
    df_0h.columns = ['0h_'+str(sta) for sta in df.columns]
    df_6h.columns = ['6h_'+str(sta) for sta in df_6h.columns]
    df_12h.columns = ['12h_'+str(sta) for sta in df_12h.columns]
    df_18h.columns = ['18h_'+str(sta) for sta in df_18h.columns]

    df_shifted = pd.concat([df_0h, df_6h, df_12h], axis=1, join='inner')
    df_shifted.dropna(inplace=True, how='any')


    nb_pc = 8
    outpath ="../../res/regional/ceof/TUV_6h12h/"
    stamod_ceof, loadings_ceof, scores_ceof ,dfs_pcs_reconstruct_ceof = do_regional_pca(df_shifted, nb_pc, outpath = outpath + "/")
    raster_val, raster_lon, raster_lat, wind_stalatlon, T_stalatlon, Q_stalatlon = get_background_and_pos(staname_t, staname_q, staname_wind)

    for npc in range(1, nb_pc+1):



        # UV
        rec_daily = dfs_pcs_reconstruct_ceof[npc].groupby(lambda t: (t.hour)).mean()

        rec_daily_0h = rec_daily.filter(regex='0h_')
        rec_daily_6h = rec_daily.filter(regex='6h_')
        rec_daily_12h = rec_daily.filter(regex='12h_')
        # rec_daily_18h = rec_daily.filter(regex='18h_')

        load_ceof = loadings_ceof.loc[npc, :]
        load_ceof_0h = load_ceof.filter(regex='0h_')
        load_ceof_6h = load_ceof.filter(regex='6h_')
        load_ceof_12h = load_ceof.filter(regex='12h_')
        # load_ceof_18h = load_ceof.filter(regex='18h_')

        plt.subplot()

        plot_daily_map_pc(rec_daily_0h.filter(regex='T_'),rec_daily_0h.filter(regex='U_'), rec_daily_0h.filter(regex='V_'),
                          wind_stalatlon, T_stalatlon, raster_lat, raster_lon, raster_val,
                    nb_pc=npc, name='0h',outpath=outpath + "/reconstruct")

        plot_map_loadings(load_ceof_0h.filter(regex='T_'), load_ceof_0h.filter(regex='U_'), load_ceof_0h.filter(regex='V_'),
                          wind_stalatlon, T_stalatlon, raster_lat,
                          raster_lon, raster_val, nb_pc=npc, name='0h',outpath=outpath+"/loadings/"+'PC' + str(npc)+'/')

        plot_daily_map_pc(rec_daily_6h.filter(regex='T_'),rec_daily_6h.filter(regex='U_'), rec_daily_6h.filter(regex='V_'),
                          wind_stalatlon, T_stalatlon, raster_lat, raster_lon, raster_val,
                    nb_pc=npc, name='06h',outpath=outpath + "/reconstruct")

        plot_map_loadings(load_ceof_6h.filter(regex='T_'), load_ceof_6h.filter(regex='U_'), load_ceof_6h.filter(regex='V_'),
                          wind_stalatlon, T_stalatlon, raster_lat,
                          raster_lon, raster_val, nb_pc=npc, name='06h',outpath=outpath+"/loadings/"+'PC' + str(npc)+'/')

        plot_daily_map_pc(rec_daily_12h.filter(regex='T_'),rec_daily_12h.filter(regex='U_'), rec_daily_12h.filter(regex='V_'),
                          wind_stalatlon, T_stalatlon, raster_lat, raster_lon, raster_val,
                    nb_pc=npc, name='12h',outpath=outpath + "/reconstruct")

        plot_map_loadings(load_ceof_12h.filter(regex='T_'), load_ceof_12h.filter(regex='U_'), load_ceof_12h.filter(regex='V_'),
                          wind_stalatlon, T_stalatlon, raster_lat,
                          raster_lon, raster_val, nb_pc=npc, name='12h',outpath=outpath+"/loadings/"+'PC' + str(npc)+'/')
        #
        # plot_daily_map_pc(rec_daily_18h.filter(regex='T_'),rec_daily_18h.filter(regex='U_'), rec_daily_18h.filter(regex='V_'),
        #                   wind_stalatlon, T_stalatlon, raster_lat, raster_lon, raster_val,
        #             nb_pc=npc, name='18h',outpath=outpath + "TUV/reconstruct")
        #
        # plot_map_loadings(load_ceof_18h.filter(regex='T_'), load_ceof_18h.filter(regex='U_'), load_ceof_18h.filter(regex='V_'),
        #                   wind_stalatlon, T_stalatlon, raster_lat,
        #                   raster_lon, raster_val, nb_pc=npc, name='18h',outpath=outpath+"TUV/loadings/"+'PC' + str(npc)+'/')

    print('Done')
