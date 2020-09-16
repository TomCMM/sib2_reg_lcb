"""
    Description
        Gap filling methodologies

    AUthor:
        Thomas
"""

import itertools


import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations


def MLR(X, Y, summary=None, corel=False, constant=True):
    """
    Description:
        Perform the multilinear regression

    INPUT
        X: dataframe of predictor
        Y: predictant
        summary; True, print the summary of the linear regression
        corel, if True return the correlation and not the parameters
    OUTPUT
        estimator object
    EXAMPLE
        X = df_adv[['TV', 'Radio']]
        y = df_adv['Sales']

        ## fit a OLS model with intercept on TV and Radio
        X = sm.add_constant(X)
        est = sm.OLS(y, X).fit()

        est.summary()
    """

    if constant:
        X = sm.add_constant(X)
    else:
        print("No Constant use in the linear regression!")

    est = sm.OLS(Y, X).fit()

    if summary == True:
        print(est.summary())
        print(est.params)

    return est


def sort_predictors_by_corr(df, selections, response, constant=True, selectionsnames=None, sort_cor=True, cor_lim=None,
                            verbose=True):
    """
    Description:
        Sort the predictors by correlation scores

    Parameters
    Return a sorted  selections by the correlation rsquared scores

    """

    len_selections = len(selections)
    scores_corel = pd.DataFrame(index=np.arange(0, len_selections), columns=['corel', 'selections', 'params',
                                                                             'selectionname'])  # correlation of each selections and variables
    print("sorting best predictors")

    for i in range(len_selections):
        #         try:
        #             print str(int((i/float(len_selections))*100))
        #         except ZeroDivisionError:
        #             pass
        selection = selections[i]
        Y = df.loc[:, response]
        X1 = df.loc[:, selection[0]]
        X2 = df.loc[:, selection[1]]

        try:
            data = pd.concat([Y, X1, X2], keys=['Y', 'X1', 'X2'], axis=1, join='outer').dropna()

            est = MLR(data[['X1', 'X2']], data['Y'], constant=constant)
            rsquared = est.rsquared

            scores_corel.loc[i, 'corel'] = rsquared
            scores_corel.loc[i, 'selections'] = selection
            scores_corel.loc[i, 'selectionname'] = selection

            if constant:
                scores_corel.loc[i, 'params'] = [est.params[0], est.params[1], est.params[2]]
            else:
                scores_corel.loc[i, 'params'] = [est.params[0], est.params[1]]

        except (ValueError, IndexError):
            print('No data to do the multilinear regression. Put correlation = 0')
            scores_corel.loc[i, 'selections'] = selection
            scores_corel.loc[i, 'selectionname'] = selection
            scores_corel.loc[i, 'corel'] = 0
            scores_corel.loc[i, 'params'] = np.nan

    if sort_cor:
        scores_corel = scores_corel.sort_values('corel', ascending=False)

    if cor_lim:
        scores_corel = scores_corel[scores_corel['corel'] > cor_lim]
    else:
        scores_corel = scores_corel[scores_corel['corel'] > 0]

    scores_corel.index = np.arange(0, len(scores_corel.index))
    selections = scores_corel['selections'].values
    params = scores_corel['params'].values

    if verbose:
        print("u" * 30)
        print("Correlation coefficient of the multilinear regression")
        print("u" * 30)
        print(scores_corel[['corel', 'selectionname']])
        print("u" * 30)
    return selections, params


def filldf(df, response, sorted_selection, params_selection, constant=True, verbose=True):
    """
    Fill a dataframe row with the others
    """
    selections_iter = iter(sorted_selection)
    params_iter = iter(params_selection)
    idxmissing = df[response][df[response].isnull() == True].index  # slect where their is missing data

    print("Filling .... ")

    while len(idxmissing) > 0:
        print("Their is  [" + str(len(idxmissing)) + "] events missing")

        try:  # Try if their is still other stations to fill with
            selection = next(selections_iter)
            param = next(params_iter)
        except StopIteration:
            print("NO MORE SELECTED STATIONS")
            break

        try:
            Y = df.loc[:, response]
            X1 = df.loc[:, selection[0]]
            X2 = df.loc[:, selection[1]]
            select = pd.concat([X1, X2], keys=['X1', 'X2'], axis=1, join='inner').dropna()
            if constant:
                newdata = param[0] + param[1] * select['X1'] + param[2] * select['X2']  # reconstruct the data
            else:
                newdata = param[0] * select['X1'] + param[1] * select['X2']  # reconstruct the data

            df.loc[idxmissing, response] = newdata.loc[idxmissing]
            idxmissing = df[response][df[response].isnull() == True].index  # slect where their is missing data
        except KeyError:
            if verbose:
                print('Selected stations ' + str(selection) + 'did not fill any events')
            else:
                pass

        except ValueError:
            if verbose:
                print('The variable ' + var + "Does not exist or no data to do the multilinear regression ")
            else:
                pass

    return df.loc[:, response]


def fill_spatial_regr(df,distance=None, by_distance=False, cor_lim=None):
    df_filled = {}
    for i, staname in enumerate(df.columns):
        print("filling station " + str(staname))
        print("remains [" + str(i) + "/" + str(len(df.columns)) + "]")

        if not by_distance:
            predictors = list(df.columns)
            predictors.remove(staname)
            selections = list(combinations(predictors, r=2))  # create all the possible combinations
            sorted_selection, params_selection = sort_predictors_by_corr(df, selections, staname, verbose=True,cor_lim=cor_lim)

        else:
            selections= getpredictors_distance(staname, distance)
            sorted_selection, params_selection = sort_predictors_by_corr(df, selections, staname, verbose=True ,cor_lim=cor_lim)


        df_filled[staname] = filldf(df, staname, sorted_selection, params_selection, verbose=True)
    df_filled = pd.DataFrame(df_filled)
    return df_filled



def getpredictors_distance( staname, distance):
    """
    Get preditors base on their distance
    The predictors are selected as following
        [1,2], [1,3], [1,4], [2,3], [2,4], [2,5], [2,6]

    """

    distfromsta = distance[staname]
    try:
        del distfromsta[staname]  # remove the station to be fill from the dataframe
    except:
        pass
    distfromsta = distfromsta.sort_values()

    stations = distfromsta.index

    sel1 = [(i, e) for i, e in zip(stations[0:-1], stations[1:])]  # selction predictors with spacing 1
    sel2 = [(i, e) for i, e in zip(stations[0:-2], stations[2:])]  # selction predictors with spacing 2

    selection= [None] * (len(sel1) + len(sel2))
    selection[::2] = sel1
    selection[1::2] = sel2

    return selection[:4]

class FillGap():
    """
    DESCRIPTION
        Bootstrap data from a station network
    INPUT
        a network object
    """

    def __init__(self, network):
        self.network = network
        self.newdataframes = {}  # contain the new dataframes

    def fillstation(self, stanames, all=None, plot=None, summary=None, From=None, To=None, by=None,
                    how='mean', variables=None, distance=None, sort_cor=True, constant=True, cor_lim=None):
        """
        DESCRIPTION
            Check every variable of every stations and try to fill
            them with the variables of the two nearest station for every time.
        INPUT
            From: From where to select the dataver
            To: when is the ending
            by: resample the data with the "by" time resolution
            sort_cor, if True sort the selected predictors stations by correlation coefficient
            plot: Plot a comparison of the old data and the new filled data
            variables: The variables of the dataframe to be filled
            summary: print the summary of the multilinear regression
            distance: It use the longitude to determine the nearest stations
                    A dataframe from Attsta dist_matrix containing the distance between each stations can be used
            cor_lim: Default, none. if Int given it will be used as a threshold to select the stations based on their correlation coefficient
        OLD CODE:
                    try:
                        # get parameters
                        data=pd.concat([Y, X1, X2],keys=['Y','X1','X2'],axis=1, join='outer').dropna()
                        params = self.MLR(data[['X1','X2']], data['Y'], summary = summary)

                        # get new fitted data
                        select = pd.concat([X1, X2],keys=['X1','X2'],axis=1, join='inner').dropna()
                        newdata = params[0] + params[1]*select['X1'] + params[2]*select['X2']

                        # Place fitted data in original dataframe
                        idxmissing = newdataframe[var][newdataframe[var].isnull() == True].index # slect where their is missing data
                        newdataframe[var][idxmissing] = newdata[idxmissing] # Fill the missing data with the estimated serie
                    except KeyError:
                        print('Data not present in all station')
                    except ValueError:
                        print('The variable '+var+ "Does not exist to do the multilinear regression ")

                        # get new fitted data
                        select = pd.concat([X1, X2],keys=['X1','X2'],axis=1, join='inner').dropna()
                        newdata = params[0] + params[1]*select['X1'] + params[2]*select['X2']

                        # Place fitted data in original dataframe
                        idxmissing = newdataframe[var][newdataframe[var].isnull() == True].index # slect where their is missing data
                        newdataframe[var][idxmissing] = newdata[idxmissing] # Fill the missing data with the estimated serie
                    except KeyError:
                        print('Data not present in all station')
                    except ValueError:
                        print('The variable '+var+ "Does not exist to do the multilinear regression ")
        """

        if all == True:
            stations = self.network.getsta([], all=True).values()
        else:
            stations = self.network.getsta(stanames)

        for station in stations:
            staname = station.getpara('stanames')

            if variables == None:
                newdataframe = station.getData(reindex=True, From=From, To=To, by=by,
                                               how=how)  # Dataframe which stock the new data of the stations
                newdataframe['U m/s'] = station.getData('U m/s', reindex=True, From=From, To=To, by=by, how=how)
                newdataframe['V m/s'] = station.getData('V m/s', reindex=True, From=From, To=To, by=by, how=how)
                newdataframe['Ua g/kg'] = station.getData('Ua g/kg', reindex=True, From=From, To=To, by=by, how=how)
                newdataframe['Theta C'] = station.getData('Theta C', reindex=True, From=From, To=To, by=by, how=how)
                variables_name = newdataframe.columns
            else:
                newdataframe = station.getData(var=variables, reindex=True, From=From, To=To, by=by,
                                               how=how)  # Dataframe which stock the new data of the stations
                variables_name = variables
            # select and sort nearest stations
            selections, selectionsnames = self.__getpredictors_distance(staname, distance)

            for var in variables_name:
                print("I" * 30)
                print("variable -> " + var)

                try:
                    selections, params = self.__sort_predictors_by_corr(station, selections, var, From, To, by, how,
                                                                        constant=constant,
                                                                        selectionsnames=selectionsnames,
                                                                        sort_cor=sort_cor, cor_lim=cor_lim)

                    selections_iter = iter(selections)
                    params_iter = iter(params)
                    #                 print newdataframe
                    idxmissing = newdataframe[var][
                        newdataframe[var].isnull() == True].index  # slect where their is missing data

                    while len(idxmissing) > 0:
                        print("Their is  [" + str(len(idxmissing)) + "] events missing")

                        try:  # Try if their is still other stations to fill with
                            selection = selections_iter.next()
                            param = params_iter.next()
                        except StopIteration:
                            print("NO MORE SELECTED STATIONS")
                            break

                        try:
                            Y = station.getData(var, From=From, To=To, by=by, how=how)  # variable to be filled
                            X1 = selection[0].getData(var, From=From, To=To, by=by,
                                                      how=how)  # stations variable used to fill
                            X2 = selection[1].getData(var, From=From, To=To, by=by,
                                                      how=how)  # stations variable used to fill

                            select = pd.concat([X1, X2], keys=['X1', 'X2'], axis=1, join='inner').dropna()

                            if constant:
                                newdata = param[0] + param[1] * select['X1'] + param[2] * select[
                                    'X2']  # reconstruct the data
                            else:
                                newdata = param[0] * select['X1'] + param[1] * select['X2']  # reconstruct the data

                            newdataframe.loc[idxmissing, var] = newdata.loc[idxmissing, var]
                            idxmissing = newdataframe[var][
                                newdataframe[var].isnull() == True].index  # slect where their is missing data


                        except KeyError:
                            print("&" * 60)
                            print('Selected stations did not fill any events')
                except ValueError:
                    print('The variable ' + var + "Does not exist or no data to do the multilinear regression ")

                    if plot == True:
                        df = pd.concat([Y, X1, X2, newdata, newdataframe[var]],
                                       keys=['Y', 'X1', 'X2', 'estimated data', 'Estimated replaced'], axis=1,
                                       join='outer')
                        self.plotcomparison(df)

            print("Their is  [" + str(len(idxmissing)) + "] FINALLY events missing")
            # Recalculate the wind direction and speed from the U an V components

            try:
                speed, dir = cart2pol(newdataframe['U m/s'], newdataframe['V m/s'])
                newdataframe['Dm G'] = dir
                newdataframe['Sm m/s'] = speed
            except ValueError:
                print
                'No wind found in the dataframe'
            except KeyError:
                print('No wind found in the dataframe')

            self.newdataframes[staname] = newdataframe

    def WriteDataFrames(self, Outpath):
        """
        DESCRIPTION
            Write the bootstraped in a file
        INPUT:
            Outpath, path of the output directory
        """

        newdataframes = self.newdataframes
        for staname in newdataframes.keys():
            fname = staname + '.TXT'
            newdataframes[staname].to_csv(Outpath + fname, float_format="%.2f")
            print('--------------------')
            print('Writing dataframe')
            print('--------------------')

    def __getpredictors_distance(self, staname, distance):
        """
        Get preditors base on their distance
        The predictors are selected as following
            [1,2], [1,3], [1,4], [2,3], [2,4], [2,5], [2,6]

        """

        distfromsta = distance[staname]
        del distfromsta[staname]  # remove the station to be fill from the dataframe
        distfromsta = distfromsta.sort_values()

        stations = self.network.getsta(distfromsta.index.values)
        #         station = self.network.getsta(staname)

        # Only 3 closest stations
        #         sel1 = [ (i,e) for i,e in zip(stations[0:2], stations[1:3])] # selction predictors with spacing 1
        #         sel2 = [ (i,e) for i,e in zip(stations[0:2], stations[2:4])] # selction predictors with spacing 2

        # Use all stations
        sel1 = [(i, e) for i, e in zip(stations[0:-1], stations[1:])]  # selction predictors with spacing 1
        sel2 = [(i, e) for i, e in zip(stations[0:-2], stations[2:])]  # selction predictors with spacing 2

        #         sel3 = [ (i,e) for i,e in zip(stations[0:-3], stations[3:])] # selction predictors with spacing 3
        #         sel4 = [ (i,e) for i,e in zip(stations[0:-4], stations[4:])] # selction predictors with spacing 4

        # Only 3 closest stations
        #         sel1names = [ (i.getpara('stanames'),e.getpara('stanames')) for i,e in zip(stations[0:2], stations[1:3])] # selction predictors with spacing 1
        #         sel2names = [ (i.getpara('stanames'),e.getpara('stanames')) for i,e in zip(stations[0:2], stations[2:4])] # selction predictors with spacing 1

        # using all stations
        sel1names = [(i.getpara('stanames'), e.getpara('stanames')) for i, e in
                     zip(stations[0:-1], stations[1:])]  # selction predictors with spacing 1
        sel2names = [(i.getpara('stanames'), e.getpara('stanames')) for i, e in
                     zip(stations[0:-2], stations[2:])]  # selction predictors with spacing 1

        #         sel3names = [ (i.getpara('stanames'),e.getpara('stanames')) for i,e in zip(stations[0:-3], stations[3:])] # selction predictors with spacing 1
        #         sel4names = [ (i.getpara('stanames'),e.getpara('stanames')) for i,e in zip(stations[0:-4], stations[4:])] # selction predictors with spacing 1

        selection = [x for x in itertools.chain.from_iterable(itertools.izip_longest(sel1, sel2)) if x]
        selectionnames = [x for x in itertools.chain.from_iterable(itertools.izip_longest(sel1names, sel2names)) if x]

        return selection, selectionnames

    def __sort_predictors_by_corr(self, station, selections, var, From, To, by, how, constant=True,
                                  selectionsnames=None, sort_cor=True, cor_lim=None):
        """
        Return a sorted  selections by the correlation rsquared scores

        """

        scores_corel = pd.DataFrame(index=np.arange(0, len(selections)), columns=['corel', 'selections', 'params',
                                                                                  'selectionname'])  # correlation of each selections and variables

        for i, (selection, selectionname) in enumerate(zip(selections, selectionsnames)):
            try:
                Y = station.getData(var, From=From, To=To, by=by, how=how)  # variable to be filled
                X1 = selection[0].getData(var, From=From, To=To, by=by, how=how)  # stations variable used to fill
                X2 = selection[1].getData(var, From=From, To=To, by=by, how=how)  # stations variable used to fill

                data = pd.concat([Y, X1, X2], keys=['Y', 'X1', 'X2'], axis=1, join='outer').dropna()

                est = self.__MLR(data[['X1', 'X2']], data['Y'], constant=constant)
                rsquared = est.rsquared

                scores_corel.loc[i, 'corel'] = rsquared
                scores_corel.loc[i, 'selections'] = selection
                scores_corel.loc[i, 'selectionname'] = selectionname

                if constant:
                    scores_corel.loc[i, 'params'] = [est.params[0], est.params[1], est.params[2]]
                else:
                    scores_corel.loc[i, 'params'] = [est.params[0], est.params[1]]

            except ValueError:
                print('No data to do the multilinear regression. Put correlation = 0')
                scores_corel.loc[i, 'selections'] = selection
                scores_corel.loc[i, 'selectionname'] = selectionname
                scores_corel.loc[i, 'corel'] = 0
                scores_corel.loc[i, 'params'] = np.nan

        if sort_cor:
            scores_corel = scores_corel.sort_values('corel', ascending=False)

        if cor_lim:
            scores_corel = scores_corel[scores_corel['corel'] > cor_lim]
        else:
            scores_corel = scores_corel[scores_corel['corel'] > 0]

        scores_corel.index = np.arange(0, len(scores_corel.index))
        selections = scores_corel['selections'].values
        params = scores_corel['params'].values

        print("u" * 30)
        print("Correlation coefficient of the multilinear regression")
        print("u" * 30)
        print(scores_corel[['corel', 'selectionname']])
        print("u" * 30)
        return selections, params

    def __MLR(self, X, Y, summary=None, corel=False, constant=True):
        """
        INPUT
            X: dataframe of predictor
            Y: predictant
            summary; True, print the summary of the linear regression
            corel, if True return the correlation and not the parameters
        OUTPUT
            estimator object
        EXAMPLE
            X = df_adv[['TV', 'Radio']]
            y = df_adv['Sales']

            ## fit a OLS model with intercept on TV and Radio
            X = sm.add_constant(X)
            est = sm.OLS(y, X).fit()

            est.summary()
        """

        if constant:
            X = sm.add_constant(X)
        else:
            print("No Constant use in the linear regression!")

        est = sm.OLS(Y, X).fit()

        if summary == True:
            print(est.summary())
            print(est.params)

        return est

    def plotcomparison(self, df):
        df.plot(subplots=True)
        df.plot(kind='scatter', x='Y', y='estimated data')
        plt.show()

