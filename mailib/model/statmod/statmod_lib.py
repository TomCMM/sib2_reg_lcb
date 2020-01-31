#===============================================================================
#     DESCRIPTION
#        Contain the statistical model to perform the downscalling
#===============================================================================

# Todo
# -> Thinking about merging other codes function into this module

#===============================================================================
# Library
#===============================================================================


# from mailib.stanet.lcbstanet

# from stadata.lib.LCBnet_lib import *
from scipy.optimize import curve_fit
import statsmodels.api as sm

from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy as np

from mailib.model.nn.structures import RNN_ribeirao, DNN_ribeirao
from mailib.toolbox.tools import get_df_lags
from mailib.esd.quantile_maping_santander import bias_correction

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from mailib.esd.quantile_mapping import qm

class StaMod():
    """
    Contain function and plot to perform Empirical Orthogonal Function
    
    PARAMETERS
        df: dataframe with a variable for each stations data
        AttSta: att_sta_object with the attribut of the stations
    """
    def __init__(self, df, AttSta):
        self.df = df
        
        self.AttSta = AttSta
        
        self.nb_PC = []
        self.eigenvalues = [] # or explained variance
        self.eigenvectors = [] # or loadings
        self.eigenpairs = [] # contain the eigen pairs of the PCA (scores and loadings)
        self.scores = pd.DataFrame([]) # contain the PC scores
        self.params_loadings =[] # contains the fit parameters for the PC loadings
        self.params_scores = [] # contains the fit parameters for the PC scores 
        self.topo_index = [] # contain the topographic index at each station point
        self.standard = False # flag to see if the input data has been standardize
        self.scores_model = {} # contain the models for the scores
        self.loadings_model = {} # contain the models for the loadings

    def pca_transform(self, nb_PC=4,remove_mean0=False,remove_mean1=False, 
                      standard = False, sklearn=False,sklearn_kernel=False, cov=True):
        """
        Perform the Principal component analysis with SKlearn
        using singular value fft
        The dataframe is standardize
        
        
        
        parameters:
            standard: default = True, standardize the dataframe
            nb_PC: default = 4, number of principal components to be used
            sklearn: if True (default=False) use svd by sklearn
            cov: if true (by default) sue the correlation matrix to perform the PCA analysis
        
        Stock in the object
            Dataframe with:
                eigenvalues
                eigenvectors
                scores
            list of vectors:
                eigenpairs
        
        NOTE:
            By default sklearn remove the mean from the dataset. So I cant use it to perform the downscalling
        
        References:
            http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html#projection-onto-the-new-feature-space
        """
        df = self.df
        self.nb_PC = nb_PC

        if remove_mean0:
            print('remove_mean0')
            df = df.subtract(df.mean(axis=0), axis='columns')

        if remove_mean1:
            print('remove_mean1')
            df = df.subtract(df.mean(axis=1), axis='index')
            print(df)
        
        if standard:
            # standardize
#             df_std = StandardScaler().fit_transform(df)
            self.standard = True
            df = (df - df.mean(axis=0)) / df.std(axis=0) # another way to standardise
        
        
        #=======================================================================
        # Sklearn
        #=======================================================================
        if sklearn:
            print("o"*80)
            print("SVD sklearn used")
            print("o"*80)

            if sklearn_kernel:
                print('sklearn_kernel')
                pca = KernelPCA(nb_PC, kernel="rbf", fit_inverse_transform=True, gamma=10)
            
            #Create a PCA model with nb_PC principal components
            else:
                pca = PCA(nb_PC)
            # fit data
            pca.fit(df)
             
            #Get the components from transforming the original data.
            scores = pca.transform(df) #  or PCs
            eigenvalues = pca.explained_variance_
            eigenvectors = pca.components_ # or loading 
             
            # Make a list of (eigenvalue, eigenvector) tuples   
            self.eigpairs = [(np.abs(self.eigenvalues[i]), self.eigenvector[i,:]) for i in range(len(self.eigenvalues))]

        #=======================================================================
        # Covariance Matrix
        #=======================================================================
        if cov:
            print("o"*80)
            print("Covariance used")
            print("o"*80)
            
            X = df.values
            cov_mat = np.cov(X.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

            scores = X.dot(eigenvectors)
            scores = pd.DataFrame(scores, columns = np.arange(1,len(df.columns)+1), index=df.index)
            eigenvalues = pd.Series(eigenvalues, index= np.arange(1,len(df.columns)+1))
            eigenvectors = pd.DataFrame(eigenvectors.T, columns=df.columns, index=np.arange(1,len(df.columns)+1))


        self.scores = scores.iloc[:, 0:nb_PC]
        self.eigenvalues =  eigenvalues#[0:nb_PC]
        self.eigenvectors =  eigenvectors[0:nb_PC]


        tot = sum(eigenvalues)
        self.var_exp = [(i / tot)*100 for i in sorted(eigenvalues, reverse=True)]


    def pca_reconstruct(self, pcs = None):
        """
        Reconstruct the original dataset with the "nb_PC" principal component
        Note:
            The idea is to reconstruct by hand to see if the downscalling is done correctly
            
        pcs: PCs to use to reconstruct the data
        """
        eigenvectors = self.eigenvectors
        scores = self.scores
        
#         df = pd.DataFrame(columns=eigenvectors.columns, index=scores.index)

        if pcs == None:
            pcs = scores.columns

        dfs_pcs = {}

        for pc in pcs:
            print(pc)
            loads = pd.concat([eigenvectors.loc[pc,:] for i in range(len(scores.loc[:,pc]))], axis=1).T # loading matrix
            loads.index = scores.loc[:,pc].index
            
            sc = pd.concat([scores.loc[:,pc] for i in range(len(eigenvectors.loc[pc,:]))], axis=1) # scores matrix
            sc.columns = loads.columns
            df = sc.multiply(loads)
            dfs_pcs[pc] = df
#         for sta in eigenvectors.columns:
#             for i, PC in enumerate(pcs):
#                 scores[PC]*eigenvectors[sta][PC]
#                 else:
#                     df[sta] = df[sta] + scores[PC]*eigenvectors[sta][PC]
#         print df
        return dfs_pcs

#     def curvfit_loadings(self, predictors=None, fit=None):
#         """
#         DESCRIPTION
#             Fit the loadings of the principal components with input independant variable
#             (In theapplication of the LCB it is most probably the altitude or its derivative)
#         RETURN
#             The parameters of the linear regression
#             predictors: topographic parameters to fit the loadings
#                     namerasters = [
#                    "8x8",
#                    "topex",
#                     "xx8_10000_8000____",
#                     "xx8_20000_18000____"
#                    ]
#             params_sep: list of parameters value to perfom 2 linear regression
#             curvfit: type of fit that you want to use, linear or poly
#         """
#
#
#         if not predictors:
#             params = ['Alt']* self.nb_PC
#
#         if not fit:
#             fit = [lin]* self.nb_PC
#
#         fit_parameters = []
#
#         for PC, row in self.eigenvectors.iterrows():
#             X = np.array(self.AttSta.getatt(self.df.keys(), predictors[PC - 1]))
#             self.topo_index.append(X)
#
#
#             popt, pcov = curve_fit(fit[PC-1], X, row)
#             fit_parameters.append([x for x in popt])
#
# #         print fit_parameters
# #         fit_parameters = np.vstack(fit_parameters)
# #         print
#         self.predictors = [pd.DataFrame(fit_parameters, index =range(1,self.nb_PC+1), columns = range(4))]
#         self.curvfit_loadings = fit
#         return self.params_loadings
#
#     def curvfit_scores(self, predictors, fit=None):
#         """
#         DESCRIPTION
#             Fit the Scores of the principal components with a variables
#         Input
#             A serie
#         Return
#             The parameters of the linear regression
#
#         """
#
#         if not fit:
#             fit = [lin]* self.nb_PC
#
#         scores = self.scores
#
#         fit_parameters = []
#         for i, predictor in enumerate(predictors):
#             predictor = predictor.dropna(axis=0, how='any')
#             score = scores.iloc[:,i]
#
#             df = pd.concat([predictor, score], axis=1, join='inner')
#
#             popt, pcov = curve_fit(fit[i-1], df.iloc[:,0], df.iloc[:,1])
#
#             fit_parameters.append([x for x in popt])
#
#         fit_parameters = np.vstack(fit_parameters)
#         self.params_scores = [pd.DataFrame(fit_parameters, index =range(1,self.nb_PC+1), columns = range(len(popt)))]
#         self.curvfit_scores = fit
#         return self.params_scores
#
#     def curvfit_predict(self, predictors_loadings, predictors_scores, params_loadings=None, params_scores=None):
#         """
#         DESCRIPTION
#             Return an 1/2d array estimated with the previously fit's parameters and predictos
#         RETURN
#             a dictionnary of 3D numpy array containing the estimated "loadings", "scores" and "reconstructed variable"
#         INPUT
#
#             predictors_loadings: 1/2d array, to be used with the loading
#                              parameters to create a loading 2d array
#             predictors_scores: Serie, to be used with the scores params to reconstruct the scores
#
#             params_loadings: (Optional) A dataframe of the parameters of the linear regression from
#                             curvfit_loadings. By default, the methods will look for params_loadings in the object.
#                             If it does not exist you should run curvfit_loadings
#             parms_scores: (Optional) Parameters dataframe from curvfit_scores.  By default, the methods will look for params_loadings in the object.
#                             If it does not exist you should run curvfit_scores
#
#         TODO
#             I should implement a way to use a pandas serie for the input "predictor instead of a number
#         """
#
#         loadings = []
#         scores = []
#         predicted = []
#
#         curvfit_loadings = self.curvfit_loadings
#         curvfit_scores = self.curvfit_scores
#
#
#         if (not params_loadings or not params_scores):
#             print 'Getting parameters'
#             params_loadings, params_scores = self.get_params()
#
#
#         # NEED TO IMPLEMENT MATRIX MULTIPLICATION!!!!!!!!!!!!!! I use to much loop
#         for PC_nb, fit_loading, fit_score, predictor_loadings, predictor_scores in zip(range(1,self.nb_PC+1),curvfit_loadings, curvfit_scores,predictors_loadings,predictors_scores ):
#             loading_est = fit_loading(predictor_loadings, *params_loadings[0].loc[PC_nb,:])
#             score_est =fit_score(predictor_scores, *params_scores[0].loc[PC_nb,:])
#
#             score = pd.concat([score_est]*len(loading_est), axis=1)
#             predict= score.multiply(loading_est)
#
#             loadings.append(loading_est)
#             scores.append( score_est)
#             predicted.append( predict)
#
#         loadings = np.array(np.dstack(loadings))
#         scores = np.array(np.dstack(scores))
#         predicted = np.array(np.dstack(predicted))
#
#
#         res = {'loadings':loadings, 'scores':scores, 'predicted': predicted}
#         return res
    
    def fit_curvfit(self, X, Y, fits=None):
        """
        fit using curvfit
        
        Parameters: 
            X, a list of dataframes
            Y, a dataframe (scores or loading to fit)
            fits, a list of list witht the function to fit
            
        """

        res_predictors_name = []
        res_fits = []
        res_fit_parameters = []
        for i in range(len(Y.keys())):

            x = X[i]
            y = Y.iloc[:,i]
            fit = fits[i]

            popt, pcov = curve_fit(fit, x.values, y.values)

            try:
                res_predictors_name.append([x.columns])
            except AttributeError:
                res_predictors_name.append([x.name])
            res_fit_parameters.append([popt])
            res_fits.append(fit)

#         fit_parameters = np.vstack(fit_parameters)
        res_predictors_name = pd.DataFrame(res_predictors_name, index =Y.keys())
        # res_predictors_name.index = Y.keys()
        res_fit_parameters = pd.DataFrame(res_fit_parameters, index =Y.keys())
        res_fits = pd.DataFrame(res_fits, index =Y.keys())

        df_models = pd.concat([res_predictors_name, res_fits, res_fit_parameters],axis=1)
        df_models.columns = ['predictor', 'model','params' ]
        df_models.index = Y.keys() #range(1,len(Y.keys())+1 )

        return df_models 
    
#     def curvfit_skill(self, df_verif, predictors_scores, metrics, params_loadings=None, params_scores=None):
#         """
#         DESCRIPTION
#             Compute bias and RMSE to assess the model performance
#         INPUT
#             df_verif: dataframe with the observed values
#             predictors: a list of pandas series which contains the predictors for the scores SHOULD NOT BE A LIST
#             metrics: sklearn metric function to be used
#                     see: http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
#                 example:
#                     metrics.explained_variance_score(y_true, y_pred)     Explained variance regression score function
#                     metrics.mean_absolute_error(y_true, y_pred)     Mean absolute error regression loss
#                     metrics.mean_squared_error(y_true, y_pred[, ...])     Mean squared error regression loss
#                     metrics.median_absolute_error(y_true, y_pred)     Median absolute error regression loss
#                     metrics.r2_score(y_true, y_pred[, ...])     R^2 (coefficient of determination) regression score function.
#         """

#         if (not params_loadings or not params_scores):
#             params_loadings, params_scores = self.get_params()
#
#         topo_index = self.topo_index
#         data = np.array([])
#         res = self.predict(topo_index, predictors_scores)
#         data = res['predicted'].sum(axis=2)
#         df_rec = pd.DataFrame(data, columns = df_verif.columns, index = predictors_scores[0].index) # should improve this
#
#         score = pd.Series()
#
#         for sta in df_rec:
#             df = pd.concat([df_verif[sta],df_rec[sta]], axis=1, join='inner')
#             df = df.dropna(axis=0)
# #             df.columns=['True', 'Pred']
#             df.plot()
#             plt.show()
#             score[sta] = metrics(df.iloc[:,0], df.iloc[:,1])
#         return score
    def get_curvfit_params(self):
        """
        DESCRIPTION
            Return the params loadings and scores. and return and error if the model has not being fitted
        """
        
        try:
            params_loadings = self.params_loadings
            params_scores = self.params_scores
        except AttributeError:
            raise AttributeError( "The model has not been fitted, run curvfit_loadings or curvfit_scores")
        
        return params_loadings, params_scores

    def plot_exp_var(self, output=None):
        """
        DESCRIPTION
            Make a plot of the variance explaine by the principal components
        """
        print("Plot explained variance")
        var_exp = self.var_exp
        cum_var_exp = np.cumsum(var_exp)

        var_exp = var_exp[:self.nb_PC]
        idxs = range(1,self.nb_PC+1)


        ax = plt.figure(figsize=(6, 4)).gca()
    
        plt.bar(idxs, var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(idxs, cum_var_exp[:self.nb_PC], where='mid',
                 label='cumulative explained variance')
        ax.set_xticks(idxs)
        ax.set_xticklabels(idxs)
        plt.ylabel('Explained variance (%)')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(True, color='0.5')
        
        if output:
            plt.savefig(output)
            plt.close()
        else:
            plt.show()

    def plot_loading(self, params_topo=None, params_fit=None, output=False, fit=None):
        """
        DESCRIPTION
            Plot the loadings in function of some parameters
        Parameters
            output: if given save the plot at the path indicated
            params: parameters of the linear regression beetween
                 loadings and independant variables 
        """

        if not params_topo:
            params_topo = ['Alt']* self.nb_PC
        
        if not fit:
            fit = [None]* self.nb_PC
        
        
        for pc_nb, param_topo, func in zip(range(1,self.nb_PC+1), params_topo, fit):
            elev_real = self.AttSta.getatt(self.df.keys(),param_topo) 
            fig, ax = plt.subplots()
            plt.scatter(self.eigenvectors.loc[pc_nb], elev_real, s=20)

            if isinstance(params_fit, pd.DataFrame):
                x = np.linspace(min(elev_real), max(elev_real),100)
                p = params_fit.loc[pc_nb,:].dropna()
                
                y = func(x, *p)

                plt.plot(y,x)
                plt.xlabel('PC'+str(pc_nb)+' loadings')
                plt.ylabel("Altitude (m)")
                plt.grid(True, color='0.5')


            for i, txt in enumerate(self.df.columns):
                ax.annotate(txt, (self.eigenvectors.loc[pc_nb][i], elev_real[i]))
            if output:
                plt.savefig(output+str(pc_nb)+'.pdf', transparent=True)

        plt.show()
    
    def plot_scores(self, predictors, params_fit =None,fit=None, output=False):
        """
        DESCRIPTION
            Make a scatter plot of the Principal component scores in function of another variables
        INPUT
            var: time serie with the same index than the dataframe sued in the pca
        """
        
        
        scores = self.scores
        
        for i, predictor, func in zip(range(0,self.nb_PC), predictors, fit):
            predictor = predictor.dropna(axis=0, how='any')
            score = scores.iloc[:,i]
            df = pd.concat([predictor, score], axis=1, join='inner')

            plt.scatter(df.iloc[:,0],df.iloc[:,1] )
  
            x = np.linspace(min(predictor), max(predictor),100)
            p = params_fit.loc[i+1,:]
            y = func(x, *p)
            
            plt.figure()
            plt.plot(x,y)
            plt.grid(True)
            plt.show()

        if output:
            plt.savefig(output)
        else:
            plt.show()
 
    def plot_scores_ts(self, output=False):
        """
        DESCRIPTION
           Plot scores time serie
        Parameters
            output: if given save the plot at the path indicated
        """
        scores = self.scores
        scores.plot(subplots=True)
        plt.xlabel('Time')
        plt.ylabel("PCs time series")
        plt.grid(True, color='0.5')
        if output:
            plt.savefig(output)
        else:
            plt.show()

    def predict_model(self, predictors_loadings, predictors_scores=None,
                      model_loadings=None, model_scores=None, model_loading_curvfit=None, model_scores_curvfit=None,
                      observed_scores = False, dnn_model_scores =None, rnn_model_scores =None, use_predict_func_loading = None,
                      nb_PC = None, qm_maps=None,obs_scores=False, idx_test=None, idx_train=None, eqm=False):

        """


        :param predictors_loadings:
        :param predictors_scores:
        :param model_loadings:
        :param model_scores:
        :param model_loading_curvfit:
        :param model_scores_curvfit:
        :param observed_scores: Use the observed scores instead of the modeled scores for the reconstruction
        :param dnn_model_scores:
        :param use_predict_func_loading:
        :param nb_PC:
        :return:
        """


#          for PC_nb, fit_loading, predictor_loadings in zip(range(1,self.nb_PC+1),fit_loadings,predictors_loadings):
# -
# +            print PC_nb
#              p = params_loadings[0].loc[PC_nb,:].dropna()
#              p = params_loadings[0].loc[PC_nb,:].dropna()
#              loading_est = fit_loading(predictor_loadings, *p)
#              loading_est = fit_loading(predictor_loadings, *p)

        loadings = []
        scores = []
        predicted = []

        for PC_nb in range(1,nb_PC+1):

            if not isinstance(obs_scores, type(False)):
                index = obs_scores.index
                y_scores = obs_scores.values
                y_scores = y_scores[:, PC_nb-1]
            else:

                # get the predictors
                x_scores = predictors_scores.loc[:, model_scores['predictor'][PC_nb]] # TOdo warning a bug on the selection of predicotors
                model_scores_pc = model_scores['model'][PC_nb]

            # get the models
            x_loadings = predictors_loadings.loc[:, model_loadings['predictor'][PC_nb]]
            model_loading_pc = model_loadings['model'][PC_nb]

            if model_scores_curvfit:
                pass
            elif dnn_model_scores:

                x_scores = pd.DataFrame(model_scores_pc.scaler_x.transform(x_scores),index=x_scores.index, columns=x_scores.columns) # scale as for training

                y_scores = model_scores_pc.predict(x_scores)
                y_scores = model_scores_pc.scaler_y.inverse_transform(y_scores).flatten()

                index = x_scores.index

            elif rnn_model_scores:

                x_scores = pd.DataFrame(model_scores_pc.scaler_x.transform(x_scores), index=x_scores.index,
                                        columns=x_scores.columns)  # scale as for training
                # preprocessing for RNN
                X_lag = get_df_lags(x_scores, lags=range(model_scores_pc.sequence_length))
                index = X_lag.index
                # Y = Y.loc[X_lag.index, :]  # get same index
                X_lag = X_lag.values.reshape(X_lag.shape[0], model_scores_pc.sequence_length, int(X_lag.shape[1] / model_scores_pc.sequence_length))

                y_scores = model_scores_pc.predict(X_lag)
                y_scores = model_scores_pc.scaler_y.inverse_transform(y_scores).flatten()

            elif not isinstance(obs_scores, pd.DataFrame):

                if model_scores_pc.k_constant == 1 :
                    x_scores = sm.add_constant(x_scores)

                if isinstance(observed_scores, pd.DataFrame): # use observed score instead of predicted scores
                    y_scores = observed_scores.loc[:,PC_nb].values
                else:
                    y_scores = model_scores_pc.predict(x_scores)
                index = x_scores.index

            # Model loadings
            if model_loading_curvfit:
                y_loading = model_loadings['model'][PC_nb](x_loadings.values, *model_loadings['params'][PC_nb])
            elif use_predict_func_loading:
                pass
            else:
                if model_loading_pc.k_constant == 1 :
                    x_loadings = sm.add_constant(x_loadings)
                y_loading = model_loading_pc.predict(x_loadings)

            if eqm:
                obs_sc= self.scores
                sim_sc = y_scores
                y_scores = bias_correction(obs_sc.loc[idx_train,PC_nb], sim_sc.loc[idx_train],
                                                sim_sc.loc[idx_test], method='eqm' ,extrapolate='constant', nbins=50)
                index = idx_test
                print('eqm')


            if qm_maps != None:

                qmap = qm_maps[PC_nb]
                old_y_scores = y_scores
                y_scores = qmap.map(old_y_scores)

            y_loading_test = np.array([y_loading for l in range(len(y_scores))])


            predict= y_loading_test * y_scores[:,None]
            loadings.append(y_loading)
            scores.append( y_scores)
            predicted.append( predict)
     
        loadings = np.array(np.dstack(loadings))
        scores = np.array(np.dstack(scores))
        predicted = np.array(np.dstack(predicted))

        # K.clear_session()
        # gc.collect()

        res = {'loadings':loadings, 'scores':scores, 'predicted': predicted,'index':index}
        return res

    def stepwise_model(self, df_PCs, predictors, lim_nb_predictors=None, constant=True, log=False, manual_selection=None):
        """
        Find best linear predictors to fit the scores
        Linear model designed by forward selection.
    
        Parameters:
        -----------
        data : pandas DataFrame with all possible predictors and response
    
        response: string, name of response column in data
        
        log: True, print a log of the score and associated candidate
    
        Returns:
        --------
        model: an "optimal" fitted statsmodels linear model
               with an intercept
               selected by forward selection
               evaluated by adjusted R-squared
        """
        models = []
        predictors_name = []
        params = []
        dic_log ={} # contain the log

        # remove columns not float
        predictors = predictors.select_dtypes(exclude=['object'])

        print("O"*10)
        if lim_nb_predictors:
            print("Number of predictors limited to " + str(lim_nb_predictors))
        print("O"*10)
        
        for column in df_PCs:
            dic_log[column] = {}

            pc = df_PCs.loc[:,column]
            remaining = set(predictors.columns)

            selected = []
            best_score, new_best_score = 0.0, 0.0

            print("=="*20)
            print("Pc"+str(column))
            print("=="*20)

            if not manual_selection:
                for i in range(lim_nb_predictors):
                    scores_with_candidates = []
                    log_pearson_with_candidates = []
                    for candidate in remaining:
                        x = predictors[selected + [candidate]]
                        if constant:
                            x = sm.add_constant(x)

                        y = pc

                        score = sm.OLS(y.values,x.values).fit().rsquared_adj
                        log_pearson_with_candidates.append([pearsonr(x.values[:,1],y.values),candidate])
                        # score = sm.OLS(y.values,x.values).fit().rsquared
                        scores_with_candidates.append([score, candidate])

                    scores_with_candidates = pd.DataFrame(scores_with_candidates, columns = ['score', "candidate"])
                    scores_with_candidates.sort_values('score', ascending=False, inplace=True)

                    log_pearson_with_candidates= pd.DataFrame(log_pearson_with_candidates, columns = ['score', "candidate"])
                    log_pearson_with_candidates.sort_values('score', ascending=False, inplace=True)

                    new_best_score, best_candidate = scores_with_candidates.iloc[0,:]

                    if log:
                        dic_log[column][str(i)] = log_pearson_with_candidates

                    if best_score < new_best_score:
                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        print(selected)
                        print(new_best_score)
                        best_score = new_best_score
            else:
                selected = manual_selection[column]

            if selected == []: # if the stepwise regression was not able to select variables to create a model
                model = np.nan
            else:
                if constant:
                    model = sm.OLS(pc.values, sm.add_constant(predictors[selected].values)).fit()
                    print(model.rsquared_adj)
                else:
                    model = sm.OLS(pc.values, predictors[selected].values).fit()
            params.append(model.params)
            models.append(model)
            predictors_name.append(selected)
        
  
        predictors_name = pd.Series(predictors_name)
        models = pd.Series(models)
        params = pd.Series(params)



        print("Saving")
        df_models = pd.concat([predictors_name,models, params],axis=1)
        df_models.columns = ['predictor', 'model', 'params']
        df_models.index = range(1,len(df_PCs.columns)+1 )
        print("Done")
        if log:
            return df_models, dic_log
        else:
            return df_models

    def skill_model(self, df_obs, pred, metrics, use_bias=None, hours=False, plot_summary=False, summary=None, mean_data=False, type='predicted'):
        """
        DESCRIPTION
            Compute bias and RMSE to assess the model performance 
        INPUT
            df_verif: dataframe with the observed values
            predictors: a list of pandas series which contains the predictors for the scores SHOULD NOT BE A LIST
            metrics: sklearn metric function to be used
                    see: http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
                example:
                    metrics.explained_variance_score(y_true, y_pred)     Explained variance regression score function
                    metrics.mean_absolute_error(y_true, y_pred)     Mean absolute error regression loss
                    metrics.mean_squared_error(y_true, y_pred[, ...])     Mean squared error regression loss
                    metrics.median_absolute_error(y_true, y_pred)     Median absolute error regression loss
                    metrics.r2_score(y_true, y_pred[, ...])     R^2 (coefficient of determination) regression score function.
            summary: True, print the mean statistics
        """
        # if (not params_loadings or not params_scores):
        #     params_loadings, params_scores = self.get_curvfit_params()

        if type == 'predicted':
            data = pred['predicted'].sum(axis=2)
        else:
            data = pred
        
        if  isinstance(mean_data, pd.core.series.Series):
            df_rec = pd.DataFrame(data, columns=df_obs.columns, index=pred['index'])  # should improve this
            df_rec = df_rec.add(mean_data,axis=0)
            df_obs = df_obs.add(mean_data,axis=0)
        else:
            df_rec = pd.DataFrame(data, columns = df_obs.columns, index =pred['index']) # should improve this
        
        if not hours:
            hours = df_rec.index.hour
            hours = sorted(hours)
            hours = list(set(hours))
            hours = [str(str(hour)+':00').rjust(5, '0') for hour in hours]

        score = pd.DataFrame(columns= df_rec.columns)

        # for hour in hours:
        for sta in df_rec:
            df = pd.concat([df_obs[sta], df_rec[sta]], axis=1, join='inner')
            # df = df.between_time(hour,hour)
            df = df.dropna(axis=0)

            if metrics == pearsonr:
                score.loc[0,sta] = metrics(df.iloc[:,0], df.iloc[:,1])[0]
            elif use_bias == True:
                bias = df.iloc[:,0] - df.iloc[:,1]
                score.loc[0, sta] = bias.mean(axis=0)
            else:
                score.loc[0, sta] = metrics(df.iloc[:, 0], df.iloc[:, 1])

        if summary:
            score.loc['Total_hours',:] = score.mean(axis=0)
            score.loc[:,'Total_stations'] = score.mean(axis=1)
            if plot_summary:
                plt.figure()
                c = plt.pcolor(score, cmap="viridis")
                cbar = plt.colorbar()
                cbar.ax.tick_params(labelsize=14) 
#                 show_values(c)
                plt.title("Validation summary")
#                 print type(score)
#                 sns.heatmap(score)
                plt.yticks(np.arange(0.5, len(score.index), 1), score.index, fontsize=14)
                plt.xticks(np.arange(0.5, len(score.columns), 1), score.columns, fontsize=14,rotation='vertical')
                plt.show()
                print(score)
        return score

    def hovermoller_combined(self, dfs_pcs_combined, stanames_LCB=None):
        
        for pc in dfs_pcs_combined.keys():
        
            var = dfs_pcs_combined[pc].loc[:,'T'].groupby(lambda x: x.hour).mean()
            U = dfs_pcs_combined[pc].loc[:,'U'].groupby(lambda x: x.hour).mean()
            V = dfs_pcs_combined[pc].loc[:,'V'].groupby(lambda x: x.hour).mean()
    
            AttSta = self.AttSta
            if not stanames_LCB:
                stanames_LCB = AttSta.get_sta_id_in_metadata(values = ['Head'])
            
            print(stanames_LCB)
            var.columns = stanames_LCB
            U.columns = stanames_LCB
            V.columns = stanames_LCB
    
            sort = AttSta.sortsta(stanames_LCB, 'Lon')
            sorted_sta = sort['stanames']
            sorted_lon = sort['metadata']
            position, time = np.meshgrid(sorted_lon, var.index)
        
            levels_contour=np.linspace(var.min().min(),var.max().max(),100)
            levels_colorbar=np.linspace(var.min().min(),var.max().max(),10)
        
            cmap = plt.cm.get_cmap("RdBu_r")
            fig = plt.figure()
    
            cnt = plt.contourf(position, time, var.loc[:,sorted_sta].values, levels = levels_contour)
            for c in cnt.collections:
                c.set_edgecolor("face")
              
            cbar = plt.colorbar(ticks=levels_colorbar)
            plt.quiver(position,time,U.values,V.values)
    #         cbar.ax.tick_params()
    #         qk = plt.quiverkey(a, 0.9, 1.05, 1, r'$1 \frac{m}{s}$',
    #                                     labelpos='E',
    #                                     fontproperties={'weight': 'bold'})
            cbar.set_label('Some Units')
            plt.ylabel(r"Hours")
            plt.xlabel( r"Longitude (Degree)")
            plt.grid(True, color="0.5")
            plt.tick_params(axis='both', which='major')
        #             plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
         
            plt.title("Principal component " + str(pc))
            plt.show()

    def skill_scores(self, scores_true, scores_pred, metrics):

        metric_scores = {}
        for i, pc_nb in enumerate(scores_true.columns):
            metric_scores[pc_nb]= metrics( scores_true.iloc[:, i], scores_pred[:, i])

        return metric_scores

    def dnn_model(self, X, Y, epochs=None, model_path=None):
        models =[]

        # Scaling
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        X = pd.DataFrame(scaler_x.fit_transform(X.values), columns = X.columns, index=X.index)

        predictor_names = []

        for pc_nb in Y.columns:
            scaler_y = MinMaxScaler(feature_range=(0, 1))

            # qt = QuantileTransformer()
            # y = qt.fit_transform(Y.loc[:, pc_nb].values.reshape(-1, 1))
            # Y_scaled = scaler_y.fit_transform(y)


            Y_scaled = scaler_y.fit_transform(Y.loc[:, pc_nb].values.reshape(-1, 1))


            print('='*100)
            model, history = DNN_ribeirao(X, Y_scaled, epochs=epochs, model_path=model_path)

            # save scalers
            model.scaler_x = scaler_x
            model.scaler_y = scaler_y

            predictor_names.append(X.columns.tolist())
            models.append(model)

        df_models = pd.DataFrame([predictor_names,models]).T
        df_models.columns = ['predictor', 'model']
        df_models.index = range(1,len(Y.columns)+1 )
        print("Done")
        print('2')

        # K.clear_session()
        # gc.collect()
        return df_models

    def rnn_model(self, X, Y, SEQUENCE_LENGTH=8, epochs=2, model_path=None):
        models =[]
        predictor_names = []
        # Scaling
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        X = pd.DataFrame(scaler_x.fit_transform(X.values), columns = X.columns, index=X.index)

        # preprocessing for RNN
        X_lag = get_df_lags(X,lags=range(SEQUENCE_LENGTH))
        Y = Y.loc[X_lag.index,:] # get same index
        X_lag = X_lag.values.reshape(X_lag.shape[0], SEQUENCE_LENGTH, int(X_lag.shape[1] / SEQUENCE_LENGTH))

        for pc_nb in Y.columns:
            scaler_y = MinMaxScaler(feature_range=(0, 1))
            Y_scaled = scaler_y.fit_transform(Y.loc[:,pc_nb].values.reshape(-1,1))

            model, history = RNN_ribeirao(X_lag, Y_scaled, epochs=epochs, model_path=model_path)

            # save scalers
            model.scaler_x = scaler_x
            model.scaler_y = scaler_y

            model.sequence_length = SEQUENCE_LENGTH
            predictor_names.append(X.columns.tolist())
            models.append(model)
        df_models = pd.DataFrame([predictor_names,models]).T
        df_models.columns = ['predictor', 'model']
        df_models.index = range(1,len(Y.columns)+1 )
        print("Done")
        # K.clear_session()
        # gc.collect()
        return df_models
