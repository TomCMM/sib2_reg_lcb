"""
Complex EOF

References:
    Hilbert transform: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html

"""

from scipy.signal import hilbert

from mailib.decomposition.pca_regional_articleIII import *

if __name__ == "__main__":

    df_t, df_q,df_u, df_v, df_tuv, df_quv, df_tquv, df_uv, staname_t, staname_q, staname_wind = get_dataframes()
    df = pd.concat([df_u,df_v], join='inner', axis=1)
    analytic_signal =hilbert(df)
    # amplitude_envelope = np.abs(analytic_signal)
    # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs) #fs = 400.0

    nb_pc = 4
    outpath ="../../res/regional/ceof/"


    df = pd.DataFrame(analytic_signal, columns= df.columns, index= df.index)
    stamod_ceof, loadings_ceof, scores_ceof ,dfs_pcs_reconstruct_ceof = do_regional_pca(df, nb_pc, outpath = outpath + "UV/")



    print('Done')
    # df = analytic_signal

