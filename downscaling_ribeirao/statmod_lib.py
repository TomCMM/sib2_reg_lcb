#===============================================================================
#     DESCRIPTION
#        Contain the statistical model to perform the downscalling
#===============================================================================

#===============================================================================
# Library
#===============================================================================
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import glob
from clima_lib.LCBnet_lib import *
import statsmodels.api as sm
# import seaborn as sns
# sns.set()
from sklearn.preprocessing import scale, StandardScaler
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
# sklearn.preprocessing.Imputer
from scipy.stats.stats import pearsonr   

from clima_lib.LCBnet_lib import *
# from clima_lib.Irradiance import Irradiance_sim_obs
import glob
from compiler.ast import Const
from sklearn import metrics
from numpy import nan
from scipy.optimize import curve_fit
import statsmodels.api as sm

from matplotlib.ticker import MaxNLocator
# LCB_Irr = Irradiance_sim_obs.LCB_Irr

from sklearn.decomposition import PCA, KernelPCA

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

    def pca_transform(self, nb_PC=4,remove_mean0=False,remove_mean1=False, standard = False, sklearn=False,sklearn_kernel=False, cov=True):
        """
        Perform the Principal component analysis with SKlearn
        using singular value decomposition
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
            print 'remove_mean0'
            df = df.subtract(df.mean(axis=0), axis='columns')

        if remove_mean1:
            print 'remove_mean1'
            df = df.subtract(df.mean(axis=1), axis='index')
            print df
        
        if standard:
            # standardize
#             df_std = StandardScaler().fit_transform(df)
            self.standard = True
            df = (df - df.mean(axis=0)) / df.std(axis=0) # another way to standardise
        
        
        #=======================================================================
        # Sklearn
        #=======================================================================
        if sklearn:
            print "o"*80
            print "SVD sklearn used"
            print "o"*80

            if sklearn_kernel:
                print 'sklearn_kernel'
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
            print "o"*80
            print "Covariance used"
            print "o"*80
            
            X = df.values
            cov_mat = np.cov(X.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
            
#             for ev in eigenvectors:
#                 np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
#                 print('Everything ok!')


            scores = X.dot(eigenvectors)



            # I think their is no need for the next step
#             eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
#             eig_pairs.sort()
#             eig_pairs.reverse()
#             
#             # Visually confirm that the list is correctly sorted by decreasing eigenvalues
#             print('Eigenvalues in descending order:')
#             for i in eig_pairs:
#                 print(i[0])
#             print eig_pairs
#             print len(eig_pairs)
#             
#             
#             eigenvalues = np.array([])
#             eigenvectors = np.array([])
# 
#             for PC in range(0,len(eig_pairs)):
#                 eigenvalues = np.append(eigenvalues,eig_pairs[PC][0] )
#                 eigenvectors = np.append(eigenvectors, [eig_pairs[PC][1]])
#             
#             print eigenvalues
#             print eigenvectors
#             print eigenvectors.shape
# 
#             eigenvectors = np.reshape(eigenvectors,(len(eig_pairs), len(eig_pairs)))
#             print eigenvectors.shape
#             
#             print X.T.shape
#             
#             scores = X.T.dot(eigenvectors)

            scores = pd.DataFrame(scores, columns = np.arange(1,len(df.columns)+1), index=df.index)
            eigenvalues = pd.Series(eigenvalues, index= np.arange(1,len(df.columns)+1))
            eigenvectors = pd.DataFrame(eigenvectors.T, columns=df.columns, index=np.arange(1,len(df.columns)+1))
  
        self.scores = scores.ix[:, 0:nb_PC]
        self.eigenvalues =  eigenvalues[0:nb_PC]
        self.eigenvectors =  eigenvectors[0:nb_PC]
            # select only first components
#             scores = 
#             eigenvalues =
#             eigenvectors =
#             
        
        
#         self.scores = pd.DataFrame(scores, columns = np.arange(0,len(df.columns)), index=df.index)
#         self.eigenvalues = pd.Series(eigenvalues, index= np.arange(0,len(df.columns)))
#         self.eigenvectors = pd.DataFrame(eigenvectors, columns=df.columns, index=np.arange(0,len(df.columns)))
  
    def fit_loadings(self, params=None, fit=None):
        """
        DESCRIPTION
            Fit the loadings of the principal components with input independant variable
            (In theapplication of the LCB it is most probably the altitude or its derivative)
        RETURN
            The parameters of the linear regression
            params: topographic parameters to fit the loadings
                    namerasters = [
                   "8x8",
                   "topex",
                    "xx8_10000_8000____",
                    "xx8_20000_18000____"
                   ]
            params_sep: list of parameters value to perfom 2 linear regression
            curvfit: type of fit that you want to use, linear or poly
        """

        
        if not params:
            params = ['Alt']* self.nb_PC

        if not fit:
            fit = [lin]* self.nb_PC

        fit_parameters = []
    
        for PC, row in self.eigenvectors.iterrows():
            X = np.array(self.AttSta.getatt(self.df.keys(), params[PC - 1]))
            self.topo_index.append(X)

            
            popt, pcov = curve_fit(fit[PC-1], X, row)
            fit_parameters.append([x for x in popt])
        
#         print fit_parameters
#         fit_parameters = np.vstack(fit_parameters)
#         print 
        self.params_loadings = [pd.DataFrame(fit_parameters, index =range(1,self.nb_PC+1), columns = range(2))]
        self.fit_loadings = fit
        return self.params_loadings

    def fit_scores(self, predictors, fit=None):
        """
        DESCRIPTION
            Fit the Scores of the principal components with a variables
        Input
            A serie
        Return
            The parameters of the linear regression
        
        """

        if not fit:
            fit = [lin]* self.nb_PC
        
        scores = self.scores
        
        fit_parameters = []
        for i, predictor in enumerate(predictors):
            predictor = predictor.dropna(axis=0, how='any')
            score = scores.iloc[:,i]

            df = pd.concat([predictor, score], axis=1, join='inner')

            popt, pcov = curve_fit(fit[i-1], df.iloc[:,0], df.iloc[:,1])

            fit_parameters.append([x for x in popt])

        fit_parameters = np.vstack(fit_parameters)
        self.params_scores = [pd.DataFrame(fit_parameters, index =range(1,self.nb_PC+1), columns = range(len(popt)))]
        self.fit_scores = fit
        return self.params_scores

    def predict(self, predictors_loadings, predictors_scores, params_loadings=None, params_scores=None):
        """
        DESCRIPTION
            Return an 1/2d array estimated with the previously fit's parameters and predictos 
        RETURN
            a dictionnary of 3D numpy array containing the estimated "loadings", "scores" and "reconstructed variable"
        INPUT

            predictors_loadings: 1/2d array, to be used with the loading
                             parameters to create a loading 2d array
            predictors_scores: Serie, to be used with the scores params to reconstruct the scores
            
            params_loadings: (Optional) A dataframe of the parameters of the linear regression from 
                            fit_loadings. By default, the methods will look for params_loadings in the object. 
                            If it does not exist you should run fit_loadings
            parms_scores: (Optional) Parameters dataframe from fit_scores.  By default, the methods will look for params_loadings in the object. 
                            If it does not exist you should run fit_scores
        
        TODO
            I should implement a way to use a pandas serie for the input "predictor instead of a number
        """

        loadings = []
        scores = []
        predicted = []
        
        fit_loadings = self.fit_loadings
        fit_scores = self.fit_scores
        
        
        if (not params_loadings or not params_scores):
            print 'Getting parameters'
            params_loadings, params_scores = self.get_params()


        # NEED TO IMPLEMENT MATRIX MULTIPLICATION!!!!!!!!!!!!!! I use to much loop
        for PC_nb, fit_loading, fit_score, predictor_loadings, predictor_scores in zip(range(1,self.nb_PC+1),fit_loadings, fit_scores,predictors_loadings,predictors_scores ):
            loading_est = fit_loading(predictor_loadings, *params_loadings[0].loc[PC_nb,:])
            score_est =fit_score(predictor_scores, *params_scores[0].loc[PC_nb,:])
            
            score = pd.concat([score_est]*len(loading_est), axis=1)
            predict= score.multiply(loading_est)
                        
            loadings.append(loading_est)
            scores.append( score_est)
            predicted.append( predict)
        
        loadings = np.array(np.dstack(loadings))
        scores = np.array(np.dstack(scores))
        predicted = np.array(np.dstack(predicted))
        
        
        res = {'loadings':loadings, 'scores':scores, 'predicted': predicted}
        return res
    
#------------------------------------------------------------------------------ 

    def pca_reconstruct(self, pcs = None):
        """
        Reconstruct the original dataset with the "nb_PC" principal component
        Note:
            The idea is to reconstruct by hand to see if the downscalling is done correctly
            
        pcs: PCs to use to reconstruct the data
        """
        eigenvectors = self.eigenvectors
        scores = self.scores
        
        df = pd.DataFrame(columns=eigenvectors.columns, index=scores.index)

        if pcs == None:
            pcs = scores.columns
        print "allo"
        print pcs

        for sta in eigenvectors.columns:
            for i, PC in enumerate(pcs):
                if i ==0:
                    df[sta] = scores[PC]*eigenvectors[sta][PC]
                else:
                    df[sta] = df[sta] + scores[PC]*eigenvectors[sta][PC]
#         print df
        return df

    def skill(self, df_verif, predictors_scores, metrics, params_loadings=None, params_scores=None):
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
        """
        if (not params_loadings or not params_scores):
            params_loadings, params_scores = self.get_params()

        topo_index = self.topo_index
        data = np.array([])
        res = self.predict(topo_index, predictors_scores)
        data = res['predicted'].sum(axis=2)
        df_rec = pd.DataFrame(data, columns = df_verif.columns, index = predictors_scores[0].index) # should improve this

        score = pd.Series()

        for sta in df_rec:
            df = pd.concat([df_verif[sta],df_rec[sta]], axis=1, join='inner')
            df = df.dropna(axis=0)
#             df.columns=['True', 'Pred']
            df.plot()
            plt.show()
            score[sta] = metrics(df.iloc[:,0], df.iloc[:,1])
        return score

    def get_params(self):
        """
        DESCRIPTION
            Return the params loadings and scores. and return and error if the model has not being fitted
        """
        
        try:
            params_loadings = self.params_loadings
            params_scores = self.params_scores
        except AttributeError:
            raise AttributeError( "The model has not been fitted, run fit_loadings or fit_scores")
        
        return params_loadings, params_scores
           
    def plot_exp_var(self, output=None):
        """
        DESCRIPTION
            Make a plot of the variance explaine by the principal components
        """
        print "Plot explained variance"
        eig_vals = self.eigenvalues
        
        print eig_vals.index
        
        
        tot = sum(eig_vals)
        var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        
        ax = plt.figure(figsize=(6, 4)).gca()
    
        plt.bar(eig_vals.index, var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(eig_vals.index, cum_var_exp, where='mid',
                 label='cumulative explained variance')
        ax.set_xticks(eig_vals.index)
        ax.set_xticklabels(eig_vals.index)
        plt.ylabel('Explained variance (%)')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(True, color='0.5')
        
        if output:
            plt.savefig(output, transparent=True)
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
            plt.scatter(self.eigenvectors.loc[pc_nb], elev_real)

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
        scores.plot()
        plt.xlabel('Time')
        plt.ylabel("PCs time series")
        plt.grid(True, color='0.5')
        if output:
            plt.savefig(output, transparent=True)
        else:
            plt.show()
                 
    def to_adas(self, var, lat, lon, alt, date, hour, outpath=None):
        """
        DESCRIPTION
            Write a file in the ARPS ADAS format 
        PARAMETERS:
            T: list, Temperature
            lat: list, Latitude
            lon: list, Longitude
            alt: list, Altitude
            date: Date of the observations, format: '%Y-%m-%d %H:%M:%S'
            
        TODO:
            Adapt for other variables than temperature
        """  
        sep=7
        nobs=len(var)

        print nobs
        id = "SA"
        
        
        if not outpath:
            outpath = '/home/thomas/surfass.lso'
        
        #------------------------------------------------------------------------------ 
        #        Write file 
        #------------------------------------------------------------------------------ 
        f_out=open(outpath, 'w')
        
        #file header
        f_out.write(" "+"{} {} {} {}{} {}\n".format(date.strftime('%d-%b-%Y'),hour.strftime('%H:%M:%S')+'.00',str(0).rjust(5),str(0).rjust(4),str(nobs).rjust(sep)*7,9999))
        
        for i in range(nobs):
            #station header
            f_out.write("{} {} {} {} {} {} {}\n".format(str(id).rjust(5),np.around(lat[i],decimals=2),str(np.around(lon[i],decimals=2)).rjust(sep),
                                                        str(np.around(alt[i],decimals=0)).rjust(5),str("SA").rjust(2),str(hour.strftime('%H%M')).rjust(10), "".rjust(8) ))
            #Data variable:line1
            f_out.write(" {} {} {} {} {} {} {} {} {}\n".format(str(np.round(var[i],decimals=1)).rjust(9),str(-99.9).rjust(6),str(-99.9).rjust(5),
                                                               str(-99.9).rjust(5),str(-99.9).rjust(5),str(-99.9).rjust(5),str(-99.9).rjust(6),str(-99.9).rjust(6),str(-99.9).rjust(6)))
            #Data variable:line2
            f_out.write("{} {} {} {} {} {} {}\n".format(str(0).rjust(6),str(-99.9).rjust(7),str(-99.9).rjust(7),str(-99.9).rjust(5),str("-99.900").rjust(7),
                                                        str(-99.9).rjust(6),str(-99).rjust(4)))
        
        #close the file
        f_out.close()
        
        print("################################################################")
        print("name of the output file: "+ outpath)
        print("Variable: "+ id)
        print("domain-> Lon(" + str(lon.min())+" to "+str(lon.max())+") Lat("+str(lat.min())+" to "+str(lat.max())+")")
        print("Date:"+ str(date) + str(hour))
        print("################################################################")
        print("Sucessful!!")
        print("################################################################")

    def to_swat(self):
        """
        DESCRIPTION
            Write the ouput in the Swat input format
        """

    def predict_model(self,predictors_loadings, predictors_scores, params_loadings=None):
        
        
        loadings = []
        scores = []
        predicted = []
        
        fit_loadings = self.fit_loadings
        scores_model = self.scores_model

        if not params_loadings :
            print 'Getting parameters'
            params_loadings, params_scores = self.get_params()


        # NEED TO IMPLEMENT MATRIX MULTIPLICATION!!!!!!!!!!!!!! I use to much loop
        for PC_nb, fit_loading, predictor_loadings in zip(range(1,self.nb_PC+1),fit_loadings,predictors_loadings):
#             print PC_nb
            p = params_loadings[0].loc[PC_nb,:].dropna()
            loading_est = fit_loading(predictor_loadings, *p)
# #             
# #             print predictors_scores
# #             print "="*100
# #             print predictors_scores.loc[:, scores_model['predictor'][PC_nb]]
#             
            score_est = scores_model['model'][PC_nb].predict(predictors_scores.loc[:, scores_model['predictor'][PC_nb]])
            score_est = np.array([score_est for l in range(len(loading_est))])
            score_est = pd.DataFrame(score_est).T
            print score_est.shape
            predict= score_est.multiply(loading_est)
# # 
# #             print score_est
# #             print len(loading_est)
# #             score = [score_est for l in range(len(loading_est))]
# 
# #             score = pd.concat([score_est for l in range(len(loading_est))], axis=1)
# #             print "C"
# #             print score
# #             print loading_est
#             predict= score* np.array(loading_est)
# #             print "D" 
#             
            loadings.append(loading_est)
            scores.append( score_est)
            predicted.append( predict)
#         
        loadings = np.array(np.dstack(loadings))
        scores = np.array(np.dstack(scores))
        predicted = np.array(np.dstack(predicted))

        
        
        res = {'loadings':loadings, 'scores':scores, 'predicted': predicted}
        return res
        
    def stepwise(self, df, lim_nb_predictors=None):
        """Linear model designed by forward selection.
    
        Parameters:
        -----------
        data : pandas DataFrame with all possible predictors and response
    
        response: string, name of response column in data
    
        Returns:
        --------
        model: an "optimal" fitted statsmodels linear model
               with an intercept
               selected by forward selection
               evaluated by adjusted R-squared
        """
        models = []
        predictors_name = []

        print "O"*10
        if lim_nb_predictors:
            print "Number of predictors limited to " + str(lim_nb_predictors)
        print "O"*10
        
        for column in self.scores:
            PCA_score = self.scores[column]
            PCA_nb = PCA_score.name

            remaining = set(df.columns)
            data = pd.concat([df, PCA_score], axis=1, join='inner')
    
            selected = []
            current_score, best_new_score = 0.0, 0.0
        
            print "=="*20
            print "Pc"+str(column)
            print "=="*20
            while remaining and current_score == best_new_score:
                scores_with_candidates = []
                for candidate in remaining:
    
                    score = sm.OLS(data[PCA_nb], data[selected + [candidate]]).fit().rsquared_adj
                    scores_with_candidates.append((score, candidate))
    
                scores_with_candidates.sort()
                best_new_score, best_candidate = scores_with_candidates.pop()

                if len(selected) < lim_nb_predictors or not lim_nb_predictors:
                    if current_score < best_new_score:
                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        print selected
                        print best_new_score
                        current_score = best_new_score
    #         formula = "{} ~ {} + 1".format(response,
    #                                        ' + '.join(selected))
            model = sm.OLS(data[PCA_nb], data[selected]).fit()
            models.append(model)
            predictors_name.append(selected)
        predictors_name = pd.Series(predictors_name)
        models = pd.Series(models)

        print "Saving"
        scores_model = pd.concat([predictors_name,models],axis=1)
        scores_model.columns = ['predictor', 'model']
        scores_model.index = range(1,self.nb_PC+1 )
        self.scores_model = scores_model 
        print "Done"
        return self.scores_model

    def skill_model(self, df_verif, res, metrics, params_loadings=None, params_scores=None,
                     plot_bias=None, hours=False, plot_summary=False, summary=None):
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
        if (not params_loadings or not params_scores):
            params_loadings, params_scores = self.get_params()

        data = res['predicted'].sum(axis=2)
        df_rec = pd.DataFrame(data, columns = df_verif.columns, index =df_verif.index) # should improve this
        
        if not hours:
            hours = df_rec.index.hour
            hours = sorted(hours)
            hours = list(set(hours))
            hours = [str(str(hour)+':00').rjust(5, '0') for hour in hours]

        score = pd.DataFrame(columns= df_rec.columns, index=hours)

        for hour in hours:
            for sta in df_rec:
                df = pd.concat([df_verif[sta], df_rec[sta]], axis=1, join='inner')
                df = df.between_time(hour,hour)
                df = df.dropna(axis=0)
                
                if plot_bias:
                    df.columns=['True', 'Pred']
                    
                    res = df['True'] - df['Pred']
    
                    res.plot()
                    plt.title(sta)
                    plt.show()
                score.loc[hour,sta] = metrics(df.iloc[:,0], df.iloc[:,1])

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
                print score
        return score

def show_values(pc, fmt="%.2f", **kw):
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center",fontsize=14, color=color, **kw)

#def piecewise_linear(x, k1, k2, x0=1100, y0=0.4):
#    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def piecewise_linear(x, k1, k2, x0, y0): # need to be improve!!!!!!!!!!!!!!!
    return np.piecewise(x, [x < 1100], [lambda x:k1*x + 0.5-k1*1100, lambda x:k2*x + 0.5-k2*1100])

def piecewise_linear2(x, k1, k2, x0, y0): # need to be improve!!!!!!!!!!!!!!!
    return np.piecewise(x, [x < 1100], [lambda x:k1*x + 0.5-k1*1100, lambda x:k2*x + 0.5-k2*1100])

def pol2(x, a, b, c):
    """
    Polynomial function
    """
    return a*x + b*x**2+c

def pol3(x, a, b, c,d):
    """
    Polynomial function
    """
    return a*x + b*x**2 +c*x**3 +d

def pol4(x, a, b, c,d,h):
    """
    Polynomial function
    """
    return a*x + b*x**2 +c*x**3+d*x**4+h

def lin(x, a, b):
    """
    linear function function
    """
    return a*x +b

def exp(x,a,b,c):    
    return a * np.exp(-b * x) +c

from scipy.misc import factorial
def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def theta(T,P):
    """
    Compute Potential temperature
    T: temperture n Degree
    P: Pressure in Hectopascale
    """
    Cp = 1004. # Specific heat at constant pressure J/(kg.K)
    R = 287. #Gaz constant ....
    P0 = 1000. #Standard pressure (Hpa)
    
    theta=T*(P0/P)**(Cp/R)
    
    return theta

def br(theta1, theta2, z1,z2):
    """
    Calcul Brunt Vaissalla frequency
    """
    g=9.81
    theta_mean = (theta1 + theta2) / 2.
    br = np.sqrt( (g / theta_mean)  * ((theta2 - theta1)/ (z2 - z1)) )
    
    return br

def froude(br, U, H):
    """
    Valley depth froude number 
    """
#     print br
#     print  br.shape
#     print br * H
#     print U 
    F = U / (br* H)  
     
    return F
    
def mean_error(y_true, y_pred):
    return (y_pred - y_true).mean()
    
    
    
    
    
