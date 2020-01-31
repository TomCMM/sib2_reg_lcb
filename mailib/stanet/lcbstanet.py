"""
    Description
        Manipulate stations networks meteorological observations

    TODO
        Remove useless methods like set Inpath or Plot LCB

"""

from __future__ import division
import matplotlib.pyplot as plt
import operator
import os
import numpy as np
import pandas as pd
import re
from scipy.spatial import distance
from mailib.toolbox.geo import PolarToCartesian


class man(object):
    """
    DESCRIPTION
        Main library of the LCB network stations database
        Contain functions to manipulate, access and calculate network stations variables
    """
    def __init(self):
        pass

    def module(self,var):
        module={
              'Es hpa':self.__Es,
              'Ws g/kg':self.__Ws,
              'Qa g/kg':self.__q,
              'Theta C':self.__Theta,
              'Ev hpa':self.__Ev,
              'Uw m/s':self.__U,
              'Vw m/s':self.__V,
              'Ta C':self.__T,
              'Td C':self.__Td,
              'Sw m/s':self.__Sm,
              'Pa H':self.__Pa,
              'Rh %':self.__Rh,
              "Rad W/m2":self.__radw
              }
        return module[var]

    def __T(self):
        """
        need for recalculating
        """
        return self.Data.loc[:,'Ta C']
#         return self.getvar('Ta C')

    def __Rh(self):
        """
        need for recalculating
        """
#         return self.getvar('Ua %')
        return self.Data.loc[:,'Rh %']

    def __Td(self):
        """
        need for recalculating
        """
        return self.Data.loc[:,'Td C']


    def __Sm(self):
        """
        need for recalculating
        """
        return self.Data.loc[:,'Sw m/s']
    
    def __Pa(self):
        """
        need for recalculating
        """
#         return self.getvar('Pa H')

    def __Es(self):
        """
        Return the vapor pressure at saturation from the Temperature
        
        Bolton formula derivated from the clausius clapeyron equation 
        Find in the American Meteorological Glaussary
        T is given in Degree celsus
        es is given in Kpa (but transform in hPa in this code)
        """
        es=0.6112*np.exp((17.67*self.getvar('Ta C'))/(self.getvar('Ta C')+243.5))*10 #hPa
        #self.__setvar('Es Kpa',es)
        return es

    def __Ws(self):
        """
        Return Mixing ratio at saturation calculated from
        the vapor pressure at saturation and the total Pressure
        """
        ws=self.getpara('E')*(self.getvar('Es hpa')/(self.getvar('Pa H')-self.getvar('Es hpa')))
        #self.__setvar('Ws g/kg',ws)
        return ws

    def __Ev(self):
        """
        Vapor pressure 
        """
        w= self.getvar('Ua g/kg')* 10**-3
        e=self.getpara('E')
        p=self.getvar('Pa H')

        Ev = (w/(w+e))*p
        return Ev

    def __q(self):
        """
        Compute the specfic humidity from the relative humidity, Pressure and Temperature
        """
        q=((self.getvar('Rh %')/100)*self.getvar('Ws g/kg'))/(1-(self.getvar('Rh %')/100)*self.getvar('Ws g/kg'))*1000
        #self.__setvar('Ua g/kg',q)
        return q

    def __Theta(self): 
        """
        Compute the Potential temperature
        """
        
        theta=(self.getvar('Ta C'))*(self.getpara('P0')/self.getvar('Pa H'))**(self.getpara('Cp')/self.getpara('R'))
        #self.__setvar('Theta C',theta)
        return theta

    def __U(self):
        """
        Return the wind in the X direction (U) in m/s
        """
        U,V = PolarToCartesian(self.getvar('Sw m/s'),self.getvar('Dw G'))
        return U

    def __V(self):
        """
        Return the wind in the Y direction (V) in m/s
        """
        U,V = PolarToCartesian(self.getvar('Sw m/s'), self.getvar('Dw G'))
        return V

    def __radw(self):
        """
        DESCRIPTION
            Input the radiation in W kj/m2 (from inmet stations)
            and return in w/m2
        """
        try: # try with the Kj/m2
            data = self.getvar('Rad kJ/m2')
            data = (data * 1000) / 3600.
        except KeyError:
            print('could not calculate Kj/m2')
        return data

    def getvarRindex(self,data):
        Initime=self.getpara('From')
        Endtime=self.getpara('To')
        Initime=pd.to_datetime(Initime)# be sure that the dateset is a Timestamp
        Endtime=pd.to_datetime(Endtime)# be sure that the dateset is a Timestamp

        newdata=data.groupby(data.index).first()# Only methods wich work

        idx=pd.date_range(Initime,Endtime,freq=self.getpara('freq'))
        newdata=newdata.reindex(index=idx)
        return newdata

    def reindex(self,data, From, To):
        """
        Reindex with a 2min created index
        """
#         Initime=self.getpara('From')
#         Endtime=self.getpara('To')
#         by = self.getpara('by')# need it to clean data hourly



        Initime=pd.to_datetime(From)# be sure that the dateset is a Timestamp
        Endtime=pd.to_datetime(To)# be sure that the dateset is a Timestamp
        newdata=data.groupby(data.index).first()# Only methods wich work

        try:
            idx=pd.date_range(Initime,Endtime,freq=self.getpara('freq')) # need it to clean data hourly
        except ValueError:
            print('Could not reindex')
        except AttributeError:
            print('Remove NatTime')
            # Drop Nat
            idx=pd.date_range(Initime,Endtime,freq=self.getpara('freq'))


        newdata=newdata.reindex(index=idx)
        return newdata

    def getData(self,var= None, every=None,net='mean', From=None,To=None,From2=None, To2=None, by=None, how = "mean" , 
                group = None ,rainfilter = None , reindex =None, recalculate =False):
        """
        DESCRIPTION
            More sophisticate methods to get the LCBobject data than "getvar"
        INPUT
            If no arguments passed, the methods will look on the user specified parameters of the station
            If their is no parameters passed, it will take the parameters by default
            
            var: list of variable name
            
            net: how to group the data from the network
    
            group: 'D':Day, 'H':hour , 'M':minute, 'MH': minutes and hours
            
            resample:
                    B       business day frequency
                    C       custom business day frequency (experimental)
                    D       calendar day frequency
                    W       weekly frequency
                    M       month end frequency
                    BM      business month end frequency
                    CBM     custom business month end frequency
                    MS      month start frequency
                    BMS     business month start frequency
                    CBMS    custom business month start frequency
                    Q       quarter end frequency
                    BQ      business quarter endfrequency
                    QS      quarter start frequency
                    BQS     business quarter start frequency
                    A       year end frequency
                    BA      business year end frequency
                    AS      year start frequency
                    BAS     business year start frequency
                    BH      business hour frequency
                    H       hourly frequency
                    T       minutely frequency
                    S       secondly frequency
                    L       milliseonds
                    U       microseconds
                    N       nanoseconds
            how: how to group the data : mean or sum
            
        """

        #=======================================================================
        # Get the default parameters
        #=======================================================================
        if From == None:
#             pass
            From=self.getpara('From')
            if From == None:
                From = self.Data.loc[pd.notnull(self.Data.index)].index[0] # To be sure that no NaT enter in the dataframe
                self.setpara('From',From)
        else:
            self.setpara('From', From) # This cause problem to get module data 

        if To == None:
            To=self.getpara('To')
            if To == None:
                To = self.Data.loc[pd.notnull(self.Data.index)].index[-1] # To be sure that no NaT enter in the dataframe
                self.setpara('To', To)
        else:
            self.setpara('To', To)
        
        if To2==None:
            pass
        else:
            pass
#             self.setpara('To2',To2)
        
        if From2==None:
            pass
        else:
            pass
#             self.setpara('From2',From2)

#         #=======================================================================
#         # Get data from the network
#         #=======================================================================
        if isinstance(self, Net):
            print("I am a network")
            if net=='mean':
                print('I am averaging over all the stations')
                panel = self.getpanel()
                self.Data = panel.mean(0)

        # Remove Nat from the data set


        #=======================================================================
        # Select the variables
        #=======================================================================
        if var == None:
            data = self.Data
        else:
            if not isinstance(var, list):
                var = [var]
            data = self.Data
            for v in var:
                if v not in data.columns or recalculate:
                    print('Variable '+v +' not in original dataset, try to recalculate ...')
                    try: 

                        data[v] = self.getvar(v, recalculate=recalculate) # calculate the variable
                    except KeyError:
                        print("Return empty time serie")
                    except TypeError:
                        print("'NoneType' object has no attribute '__getitem__'")

                # else:
                #     data = self.Data

        #=======================================================================
        # Reindex if needed
        #=======================================================================
        if reindex == True:
            if not From2:
                data = self.reindex(data, From,To)
            else:
                raise 
                print("Need to implement the from2 to 2 reindexing")
#                 data = self.reindex(data.append(data[From2:To2]), From,To)
        else:
            if not From2:
                data = data[From:To]
            else:
                data = data[From:To].append(data[From2:To2])

        #=======================================================================
        # Apply a filter for the rain
        #=======================================================================
        if rainfilter == True: # need to be implemented in a method
            data=data[data['Rc mm'].resample("3H",how='mean').reindex(index=data.index,method='ffill')< 3]
            if data.empty:
                raise ValueError(" The rainfilter removed all the data -> ")



        #=======================================================================
        # Resample
        #=======================================================================


        if by != None:
#             data=data.resample(by, how = how)
            if how=='sum':
                data=data.resample(by).sum()
#                 data=data.resample(by,how=lambda x: x.values.sum()) # This method keep the NAN
            if how=='mean':
                data=data.resample(by).mean() # This method keep the NAN

#         print("data
        #===============================================================================
        # Select the period
        #===============================================================================

        data = data[From:To] # I should remove what is before

        #===============================================================================
        # Group
        #===============================================================================
        
        if group:
            if how == "sum":
                if group == 'M':
                    data=data.groupby(lambda t: (t.month)).sum()
        
                if group == 'D':
                    data=data.groupby(lambda t: (t.day)).sum()
        
                if group == 'H':
                    data=data.groupby(lambda t: (t.hour)).sum()

                if group == 'T':
                    data=data.groupby(lambda t: (t.minute)).sum()
                if group == "TH":
                    data=data.groupby(lambda t: (t.hour,t.minute)).sum()
        
            elif how == 'mean': # IT was set as "none"
                if group == 'M':
                    data=data.groupby(lambda t: (t.month)).mean()
        
                if group == 'D':
                    data=data.groupby(lambda t: (t.day)).mean()
        
                if group == 'H':
                    
                    data=data.groupby(lambda t: (t.hour)).mean()
        
                if group == 'T':
                    data=data.groupby(lambda t: (t.minute)).mean()
        
                if group == "TH":
                    data=data.groupby(lambda t: (t.hour,t.minute)).mean()

        #=======================================================================
        # Resample by every
        #=======================================================================
        if every == "M" and var != None:
            if group:
                raise AttributeError('Option "every" can not work with the function group')
            else:
                data=data.resample(by, how = how) # I make it again just in case
                data['index'] = data.index
                data['index'] = data['index'].map(lambda x: x.strftime('%Y-%m'))

                data['days'] = data.index.day
                data = pd.pivot_table(data, index = ['days'], columns=['index'], values=var)

        if every == "D" and var != None:
            if group:
                raise AttributeError('Option "every" can not work with the function group')
            else:
                data=data.resample(by, how = how) # I make it again just in case
                data['index'] = data.index
                data['index'] = data['index'].map(lambda x: x.strftime('%Y-%m-%d'))

                data['hours'] = data.index.hour
                data = pd.pivot_table(data, index = ['hours'], columns=['index'], values=var)


        #=======================================================================
        # Return the needed variable
        #=======================================================================
        if var == None:
            return data
        else:
            try:
                return data[var]
            except:
                print("Return empty vector")
                df_ = pd.DataFrame(index=data.index, columns=[var])
                df_.fillna(np.nan)
                return df_[var]
    
    def getvar(self,varname,From=None,To=None,rainfilter=None, recalculate=False):
        """
        DESCRIPTION
            Get a variables of the station
        INPUT:
            varname: string
            Recalculate: if True, recalculate the variable
        """
        print("Getting variables" + varname)
        if From == None:
            From = self.getpara('From')
        if To == None:
            To = self.getpara('To')
        
        
        if recalculate:
            try:
                print('Recalculate the variable: '+ varname)
                var_module=self.module(varname)
                data_module=var_module()[From:To]
                data_module=self.getvarRindex(data_module)# Reindex everytime to ensure the continuity of the data

                return data_module
            except KeyError:
                varindex=self.Data[From:To].index
                df_ = pd.DataFrame(index=varindex, columns=[varname])
                df_.fillna(np.nan)
                var=self.getvarRindex(varname,df_[varname])
                print('This variable cant be calculated->  '+ varname)
                return  var
        else:
            try: # Try to get directly the variable
                var=self.Data[varname][From:To]
                var=self.getvarRindex(var)# Reindex everytime to ensure the continuity of the data
                return var
            except KeyError: # The variable is not in the dataframe
                try: # try to calculate it with the modules
                    print('Try to calculate the variable: '+ varname)
                    var_module=self.module(varname)
                    data_module=var_module()[From:To]
                    data_module=self.getvarRindex(data_module)# Reindex everytime to ensure the continuity of the data
                    self.Data[varname] = data_module
                    return data_module
                except KeyError:
                    print('Their is no module to compute this variables return empty Serie')
                    varindex=self.Data[From:To].index
                    df_ = pd.DataFrame(index=varindex, columns=[varname])
                    df_.fillna(np.nan)
                    var=self.getvarRindex(df_[varname])
                    return  var

        if rainfilter == True:
            return self.__rainfilter(var)
    
#===============================================================================
# Station
#===============================================================================

class Att_Var(object):
    """
    DESCRIPTION
        Contains variables attributs e.g. longname
        perform variable mapping between stations
    NOTE
        In the futur when I create this kind of class to access the attribut
        I should make them general.
        Because I already created this kind of class called "att_sta" but it wasn't too general
    """
    def __init__(self):
        self._attname = ['longname']
        self._att = {
                    # Temperature
                    'Ta C':{'longname': 'Temperature at 2m (C)','longname_latex':'Temperature $(C^{\circ})$'},
                     'Theta C':{'longname':'Potential temperature (C)'},

                     # humidity
                     'Rh %':{'longname':'Relative humidity (%)'},
                     'Qa g/kg':{'longname':'Specific humidity (g/kg)','longname_latex':'Specific humidity $(g.kg^{-1})$'},
                     'Td C':{'longname':'Dew point temperature (C)'},

                     # Wind
                     'Sw m/s':{'longname':'wind speed  at 2m (m/s)'},
                     'Sw10m m/s':{'longname':'wind speed at 10m (m/s)'},
                    'Swmax m/s':{'longname':'Maximum wind speed (m/s)'},

                     'Dw G':{'longname':'Wind direction (degree)'},
                     'Uw m/s':{'longname':'Zonal wind  (m/s)'},
                     'Vw m/s':{'longname':'Meridional wind  (m/s)'},

                     # Other
                     'Pr mm':{'longname':'Accumulated precipitation (mm)'},
                     'Pa H':{'longname':'Pressure Hectopascale (H)'},

                     # Fluxes and cloud cover
                     'Rad w/m2':{'longname':'Solar radiation (W/m2)'},
                    'Rad MJ/m2':{'longname':'Solar radiation (MJ/m2)'},
                    'H J':{'longname':'Heat Flux (W/m2)'},
                    'Tr %':{'longname':'global transmission fraction (%)'},
                    'Br %':{'longname':'duration of brightness  (hours)'},

                    # Soil
                    'Ts1 C': {'longname':'Soil Temperature (C) level1'},
                    'Ts2 C': {'longname':'Soil Temperature (C) level2'},
                    'Ts3 C': {'longname':'Soil Temperature (C) level3'},
                    'Rhs %':{'longname':'Soil humidity fraction'},

                    #River flow
                    "Q m3/s" : {'longname':'River discharge (m3/s)'},

                     # Extra
                     'Bat mV':{'longname':'Battery millivolt (mV)'}
                     }


        self.net_var_mapping = {
            # Weather network
            'inmet':{'temp':'Ta C','ws':'Sw m/s','wd':'Dw G','ws_max':'Swmax m/s','press':'Pa H','rh':'Rh %','rad':'Rad kj/m2','pcp':'Pr mm'},
            'metar':{'wind_speed':'Sw m/s','wind_dir':'Dw G','t_air':'Ta C','t_d':'Td C'},
            'iac':{'windSpeed2m':'Sw m/s','windSpeed10m':'Sw10m m/s','windDir':'Dw G','rad1':'Rad w/m2',
                   'flxHeat': 'H J','ur':'Rh %','tAir':'Ta C','tSoil1':'Ts1 C','tSoil2':'Ts2 C','tSoil3':'Ts3 C',
                   'press':'Pa H','soilMoist':'Rhs %','pcp':'Pr mm'},
            'iag':{'irr':'Rad MJ/m2', 'tr_glob':'Tr %','brightness':'Br %','temp':'Ta C','rh':'Rh %','pcp':'Pr mm', 'ws_med':'Dm G'},
            'cge':{'Plu_mm':'Pr mm','UR_%':'Rh %','Press_mb':'Pa H','Tar_C':'Ta C','BatV':'Bat mV'},
            'cetesb':{'temp':'Ta C','rad':'Rad W/m2', 'humid':'Rh %', 'ws_ms':'Sm m/s', 'wd_o':'Dw G', 'press':'Pa H'},
            'ribeirao':{'Ua %':'Rh %','Sm m/s':'Sw m/s','Dm G':'Dw G','Rc mm':'Pr mm'},

            # Rain gauges
            'cemaden':{'pcp':'Pr mm'},
            'daee_plu':{'pcp':'Pr mm'},
            'ana_plu':{'chuva':'Pr mm'},

            # River discharge
            'daee_flu':{'flow_avg':'Q m3/s'},
            'ana_flu':{'flow_avg':'Q m3/s'},
        }


    def showatt(self):
        print(self._attname)
    
    def getatt(self,object, attnames):
        """
        INPUT
            staname : list. name of the stations 
            att : scalar. name of the attribut
        Return 
            return attributs in a list
        TODO
            just return a dictionnary
        """

        att = [ ]
        if type(attnames) != list:
            try:
                if type(attnames) == str:
                    attnames = [attnames]
            except:
                raise
        
        for attname in attnames:
            try:
                att.append(self._att[object][attname])
            except KeyError:
                
                raise KeyError('The parameter' + attname + ' do not exist')
        return att


class Att_Sta(object):
    """
    DESCRIPTION
        -> Class containing metadata informations about the climatic network in Extrema
        -> Contain methods to manipulate the network metadata
    params:
        Path_att: path of thecsv file with all the metadata
    """
    def __init__(self, path_att="../../database_metadata.csv"):
        # type: (object) -> object

        print("**"*10)
        print('METADATA FILE USED:')
        print(path_att)
        print("**"*10)

        self.path_att = path_att
        self.attributes =pd.read_csv(path_att, index_col=0, sep =',',encoding='latin1')

        for col in self.attributes.columns:
            try:
                self.attributes[col]  = self.attributes[col].apply(pd.to_numeric)
            except ValueError:
                print('Did not convert column: %s to numeric',(col))

        self.attributes.index = self.attributes.index.astype(str)

    def addatt(self, df = None, path_df=None):
        """
        DESCRIPTION
            add a dataframe of station attribut to the already existing 
        """
        attributes = self.attributes
        if path_df:
            df = pd.read_csv(path_df, index_col=0)

        newdf = pd.concat([attributes, df], join='inner', axis=1)

        self.attributes = newdf

    def sortsta(self,sta,latlon):
        """
        DESCRIPTION:
            Return a sorted list of the Latitude or longitude of the given station names
        INPUT: sta : name of the station
               latlon:  'lat' or 'lon'
        RETURN: {stations_name,positions} sorted dictionnary
        pos={}
        """
        metadata=[]
        staname=[]
        pos={}
        for i in sta:
            pos[i]=self.getatt(i, latlon)
        sorted_pos = sorted(pos.items(), key=operator.itemgetter(1))
        for i in sorted_pos:
            metadata.append(i[1])
            staname.append(i[0])
        return {'stanames':staname,'metadata':metadata}

    def get_sta_id_in_metadata(self, values=None, all=None, params=None):
        """

        Return: list, stations id in the  metadata file selected with user parameter
        all: True, return all stations in parameters
        Params: Param_min, Param_max
                return the stations in between the param_min and params_max
        
        """

         
        if all:
            return self.attributes.index
        
        if type(values) is not list:
            raise TypeError('Must be a list')
    
        sta = [ ]
        
        if values ==[]:
            sta = self.attributes.index
        
        else: 
            for staname in self.attributes.index:
                l = []
                for k in values:
                    if k in self.attributes.loc[staname].values:
                        l.append(True)

                if len(l) == len(values): # I really dont know why using all does not work
                    sta.append(staname)

        if params:
            for param in params.keys():
#                 if sta !=[]:
                attribut = self.attributes.loc[sta,:]

                if param in self.attributes.columns.values:

                    sta = attribut[(attribut[param]> params[param][0]) & (attribut[param]< params[param][1])]
                    sta = sta.index.values.tolist()
        return sta

    def setatt(self,staname, newatt, newvalue):
        """
        DESCRIPTION
            insert a new value of a new attribut corresponding at the specific station
        """
        try:
            self.attributes.loc[staname, newatt] = newvalue
        except KeyError:
             print("Their is no station id: %s in the metadata file "% (staname, self.path_att))

    def showatt(self):
        print(self.attributes)

    def getatt(self,stanames,att):
        """
        INPUT
            staname : list. name of the stations 
            att : scalar. name of the attribut
            all: return the entire dataframe
        Return 
            return attributs in a list
        TODO
            just return a dictionnary
        """


        staatt = [ ]
        if type(stanames) != list:
            try:
                if type(stanames) == str:
                    stanames = [stanames]
            except:
                raise

        for staname in stanames:
            staatt.append(self.attributes.loc[staname,att])

        return staatt

    def set_filepath(self, stanames=None, file_path= None):
        """
        Check if  the stations id is actually in the database and if it
        the case, add the path in rhe metadata
        Associate Files to stations network id


        INPUT
            String with the directory path of the station files
        OUTPUT
            
        EXAMPLE
            InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
            AttSta = att_sta()
            AttSta.set_filepath(InPath)
        """

        extension = '.csv'


        not_in_database =[]
        for staname in stanames:

            file = file_path + staname + extension
            assert str(file)
            try:
                assert os.path.exists(file), str(file)
                self.setatt(staname, 'InPath', file)
            except AssertionError:
                print('%s not in database'% (staname + '.csv'))

    def dist_sta(self,couples, formcouples = None):
        """
            DESCRIPTION
                Calculate the distance between two points
            INPUT
                list of couple of station names 
                Couple = TRUE => will form the list of couple from 
            OUTPUT
                return a list of the distance in kilometer (
            EXAMPLE
                couple = [['C15','C04']]
                dist_sta(couple)
                
                stationnames = ['C15','C04','C05']
                dist_sta()stationnames, formcouple=True)
        """

        def rearange(sortedlist):
            """
            DESCRIPTION
                Rearange the stations name list to form couple
                 which are the input of the att_sta.dist_sta() 
            """
            couple = [ ]
            for ini,end in zip(sortedlist[:-1],sortedlist[1:]):
                couple.append([ini,end])
            return couple
        
        if formcouples == True:
            if isinstance(couples,list):
                couples = rearange(couples)
            else:
                raise TypeError('Must be a list !')

        # make sure we have a list
        if isinstance(couples,list):
            pass
        else:
            couples = [couples]
    
        # make sure we have a list of list
        if all(isinstance(i, list) for i in couples):
            pass
        else:
            couples = [couples]
    
        dist = [ ]
        for sn in couples:
            lat1 = self.getatt(sn[0],'lat')[0]
            lon1 = self.getatt(sn[0],'lon')[0]
            pos1 = (lat1,lon1)
            
            lat2 = self.getatt(sn[1],'lat')[0]
            lon2 = self.getatt(sn[1],'lon')[0]
            pos2 = (lat2,lon2)
            
            dist.append(vincenty(pos1,pos2).km)
    
        return dist

    def dist_matrix(self, stanames):
        """
        DESCRIPTION
            Return, dataframe with the distance matrix of the given stations
        IMPORTANT NOTE:
            This function return the euclidian distance based on latitude longitude
            It introduce error, locations should be converted in meters before
            but for the local scale I guess it is not a problem
        """
        lats = self.getatt(stanames, 'lat')
        lons = self.getatt(stanames, 'lon')
        coords = []
        coords  = [(lat, lon) for lat,lon in zip(lats, lons)]
        dist = distance.cdist(coords, coords, 'euclidean')
        df_dist = pd.DataFrame(dist, index=stanames,columns=stanames)
        
        return df_dist
        
    def to_csv(self, outpath, params_out=None, stanames=None):
        """
        Save the dictionnary of parameter as a dataframe
        """
        df = pd.DataFrame(self.attributes)
        df = df.T
        if params_out:
            df = df[params_out]
        if stanames:
            df = df.loc[stanames,:]

        df.to_csv(outpath)

    def getsta_in_shape(self,shapefile):
            pass

class Station(man):
    """
    DESCRIPTION
        Contain station data

    PROBLEM
        SHould calculate the specific variables only when asked
    """

    def __init__(self,InPath, net='ribeirao', clean=True, att_sta=None, att_var =None):
        """
        PROBLEM
            on raw file the option "Header=True is needed"
        SOLUTION
            self.rawData=pd.read_csv(InPath,sep=',',index_col=0,parse_dates=True)
        """

        self.para={
        }


        if att_var == None:
            self.att_var = Att_Var()
        else:
            self.att_var = att_var

        self.Data = self.__read(InPath, net=net, clean=clean)
        

        self.paradef={
                    'Cp':1004, # Specific heat at constant pressure J/(kg.K)
                    'P0':1000,#Standard pressure (Hpa)
                    'R':287,#constant ....
                    'Kelvin':272.15, # constant to transform degree in kelvin
                    'E':0.622,
#                     'By':'2min',
                    'To':None,
                    'From':None,
                    'freq':'1H'
#                     'group':'1H',
        }

        if att_sta == None: # add attributes to the stations
            att_sta = Att_Sta()
            self.__poschar(att_sta)
        else:
            self.__poschar(att_sta)

    def __read(self,InPath,net, clean=True):
        """
        read the different type of network
        and load the different parameters
        params:
            net: the network than you wnat to read
            clean if the data are clean or not
        """
        self.setpara("InPath",InPath)
        self.setpara("dirname", os.path.dirname(InPath))
        self.setpara("filename", os.path.basename(InPath))

        # TODO put the characteristics of the network in a dictionnary

        if net =='ribeirao':
            df = pd.read_csv(InPath,sep=',',index_col=0,parse_dates=True)
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('By','H')
            
        if net == "Sinda":
            if clean:
                df = pd.read_csv(InPath,index_col=0,parse_dates=True)
            else:
                df = pd.read_csv(InPath, error_bad_lines=False,skiprows=1, index_col=0, parse_dates=True)
                
                mapping = {
                            'Pluvio (mm)':'Pr mm',
                            'TempAr (oC)': 'Ta C',
                            'TempMax (oC)': 'Tamin C',
                            'TempMin (oC)':'Tamax C',
                            'UmidRel (%)': 'Rh %',
                            'VelVento10m (m/s)': 'Sm m/s',
                            'DirVento (oNV)': 'Dm G'
                            }
                
                cols = []
                for f in df.columns:
                    col = re.sub(r'[^\x00-\x7F]+',' ', f)
                    if col in mapping.keys():
                        col = mapping[col]
                    cols.append(col)
                df.columns = cols
            df.index = df.index - pd.Timedelta(hours=3) # UTC
#             df.index = df.index - pd.Timedelta(hours=3) # become it make the mean the hours after
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('by','3H')

        if net =='inmet':
            if clean:
                try:
                    df = pd.read_csv(InPath,index_col=0,parse_dates=True, error_bad_lines=False)
                except UnicodeDecodeError:
                    df = pd.read_csv(InPath,index_col=0,parse_dates=True,encoding='latin-1', error_bad_lines=False)
                    # not that clean
                    idx = pd.to_datetime(df.index, errors='coerce')
                    idx = df.index[~idx.isnull()]
                    df = df.loc[idx,:]
                    df.index = pd.DatetimeIndex(df.index)

            else:
                df = pd.read_csv(InPath,parse_dates=[[1,2,3,4]],index_col=0, keep_date_col=False, delim_whitespace=True, header=None, error_bad_lines=False)
                df.columns = ['ID', "Ta C", 'Tamax C', 'Tamin C', 'Rh %','Uamax %','Uamin %', 'Td C',
                               'Td max', 'Td min', 'Pa H', 'Pamax H', 'Pamin H', 'Sm m/s', 'Dm G',
                               'Smgust m/s', 'Rad kJ/m2', 'Pr mm' ]
                del df['ID']




            df.index = df.index - pd.Timedelta(hours=3) # UTC Once it is clean it is 
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('by','1H')

        if net == 'iac':
            if clean:
                df = pd.read_csv(InPath,index_col=0,parse_dates=True)

            else:
                def parse(yr, yearday, hrmn):
                    # transform 24 into 00 of the next day
                    if hrmn[0:2] == "24":
                        yearday = str(int(yearday)+1)# this might give a problem at the end of the year
                        hrmn ="00"+ hrmn[2:]
    
                    if hrmn == '100':
                        hrmn = '0100'
                    
                    if hrmn =='200':
                        hrmn = '0200'
                    date_string = ' '.join([yr, yearday, hrmn])

                    return pd.datetime.strptime(date_string,"%Y %j %H%M")
    
                df = pd.read_csv(InPath, parse_dates={'datetime':[1,2,3]},date_parser=parse, index_col='datetime', header=None, error_bad_lines=False)

                if len(df.columns) == 8:
                    print("type2")
                    df.columns = ['type', 'Sm m/s', "Dm G", 'Rad2 W/m2', 'Ta C', 'Rh %','???', 'Pr mm']
                else:
                    df.columns = ['type', 'Sm m/s', 'Sm10m m/s', "Dm G", 'Rad W/m2', 'Rad2 W/m2', 'FG W/m2', 'Rh %',
                               'Ta C', 'Tasoil1 C', 'Tasoil2 C', 'Tasoil3 C', 'Pa H', 'SH %', 'Pr mm']
                
                # replace missing value with Nan
                df.replace(-6999, np.NAN, inplace=True)
                df.replace(6999, np.NAN, inplace=True)
                df.replace('null', np.NAN, inplace=True)
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('by','1H')

        if net == 'svg':
            df = pd.read_csv(InPath, index_col=0, parse_dates=True )
#             df.index = df.index + pd.Timedelta(hours=2) # UTC Once it is clean it is 
            df.columns = ['Ta C', "Rh %", 'Pa kpa', 'Sm m/s', 'Dm G', 'Pr mm', 'Rad Wm/2']
            self.setpara("stanames", 'svg')
            self.setpara('by','30min')
        
        if net == 'peg':
            df = pd.read_csv(InPath, index_col=1, header=None, parse_dates=True)
#             df.index = df.index + pd.Timedelta(hours=2) # UTC Once it is clean it is 
            df.columns = ['i','Ta C']
            del df['i']
            self.setpara("stanames", 'peg')
            self.setpara('by','30min')
        
        if net =='metar':
            df = pd.read_csv(InPath, index_col=0, parse_dates=True )
            df.sort_index(inplace=True)
            df.index = df.index - pd.Timedelta(hours=3) # UTC
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('by','1H')
            
        if net =='cetesb':
            if clean:
                df = pd.read_csv(InPath, index_col=0, parse_dates=True )
                df.sort_index(inplace=True)
                self.setpara("stanames", os.path.basename(InPath)[:-4])
                self.setpara('by','1H')    
            else:
                df = pd.read_csv(InPath, index_col=0, parse_dates=True )
                mapping = {
                'UMID':'Rh %',
                'PRESS': 'Pa H',
                'WS': 'Sm m/s',
                'WD':'Dm G',
                'TEMP': 'Ta C'
                }
                
                cols = []
                for f in df.columns:
                    col = re.sub(r'[^\x00-\x7F]+',' ', f)
                    if col in mapping.keys():
                        col = mapping[col]
                    cols.append(col)
                df.columns = cols
                
#                 df.columns = ['Rh %', 'Sm m/s', 'Dm G', 'Ta C']
                df.sort_index(inplace=True)
                self.setpara("stanames", os.path.basename(InPath)[:-4])
                self.setpara('by','1H')    

        if net=='pcd':
            df = pd.read_csv(InPath, index_col=0, parse_dates=True )
            df = df.convert_objects(convert_numeric=True)
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('by','1H') 

        if net == 'cge':
            if clean:
                df = pd.read_csv(InPath,index_col=0,parse_dates=True)
            df.index = df.index - pd.Timedelta(hours=3) # UTC Once it is clean it is
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('by','1H')

        if net == 'iag':
            if clean:
                df = pd.read_csv(InPath,index_col=0,parse_dates=True)
            # df.index = df.index - pd.Timedelta(hours=3) # UTC Once it is clean it is
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('by','1H')

        if net == 'ana_flu':
            if clean:
                df = pd.read_csv(InPath,index_col=0,parse_dates=True)
            # df.index = df.index - pd.Timedelta(hours=3) # UTC Once it is clean it is
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('by','1D')

        if net == 'ana_plu':
            if clean:
                df = pd.read_csv(InPath,index_col=0,parse_dates=True)
            # df.index = df.index - pd.Timedelta(hours=3) # UTC Once it is clean it is
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('by','1D')

        if net == 'daee_plu':
            if clean:
                df = pd.read_csv(InPath,index_col=0,parse_dates=True)
            # df.index = df.index - pd.Timedelta(hours=3) # UTC Once it is clean it is
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('by','1D')

        if net == 'daee_flu':
            if clean:
                df = pd.read_csv(InPath,index_col=0,parse_dates=True)
            # df.index = df.index - pd.Timedelta(hours=3) # UTC Once it is clean it is
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('by','1D')

        if net == 'cemaden':
            if clean:
                df = pd.read_csv(InPath,index_col=0,parse_dates=True)
            # df.index = df.index - pd.Timedelta(hours=3) # UTC Once it is clean it is
            self.setpara("stanames", os.path.basename(InPath)[:-4])
            self.setpara('by','1D')

        df.rename(columns = self.att_var.net_var_mapping[net],inplace=True) # mapping variables
        
        return df
            
    def __poschar(self, att_sta):

        attributes = att_sta.attributes
        try:
            for att in attributes:
                self.setpara(att,att_sta.getatt(self.getpara('stanames'), att))
        except KeyError:
            print('This stations doesnt have characteristics')

    def __stationname(self):
        InPath=self.getpara("InPath")

    def showvar(self):
        for i in self.Data.columns:
            print(i)


    def showpara(self):
        print(self.para)

    def __rainfilter(self,var):
        """
        Return the index of the series whithout daily rain
        Valleynet.Data['Pr mm'].resample("D",how='mean').reindex(index=Valleynet.Data.index,method='ffill')
        """
        Rain=self.getvar(self,'Pr mm').resample("D",how='mean')
        return var[Rain<0.1]

    def __setvar(self,varname,data):
        try:
            self.Data[varname]=data
        except:
            print('Couldnt add variable in the data'+ varname)

    def getpara(self,paraname):
        try:
            val=self.para[paraname]
            return val
        except KeyError:
#             print('Parameter by default used '+ paraname)
            try:
                val=self.paradef[paraname]
                return val
            
            except KeyError:
                print('this parameter dosenot exist '+ paraname)
                raise

    def setpara(self,name,value,key = None, append = None):
        """
        DESCRIPTION
            Set the parameter of the LCB object
        INPUT
            name : name of the parameter
            value: value of the parameter
            keys: if Keys != None then it will be the key of a dictionnary
            append: if == None then the newvalue overwrite the old one
        """

        if name == 'To' or name == 'From': # convert in datetime format
            value=pd.to_datetime(value)


        if append == None:
            if key == None:
                self.para[name]=value
            else:
                self.para[name] = {key:value}
        else:
            try:
                if key == None:
                    oldpara = self.para[name]
                    oldpara.append(value)
                else:
                    oldpara = self.para[name]
                    oldpara[key] = value
                self.para[name] = oldpara
            except KeyError:
                # To initialise the parameter
                if key == None:
                    self.para[name]=value
                else:
                    self.para[name] = {key:value}

    def __deregister(self, net):
        self.my_net = None

    def __register(self, net):
        self.my_net = net
        print(self.my_net)

    def report(self):
        if self.my_net:
            print("My net: ",self.my_net)
        else:
            print("i dont belong to any net")

class Net(Station):
    def __init__(self, att_sta=None, att_var=None):
        if att_sta == None:
            self.att_sta = Att_Sta()
        else:
            self.att_sta = att_sta

        if att_var == None:
            self.att_var = Att_Var()
        else:
            self.att_var = att_var

        self.min = None
        self.max = None
        self.mean = None
        self.Data=None # Which is actually the mean 
        self.para={
        }
        self.paradef={
                    'Cp':1004,# could be set somwhere else # Specific heat at constant pressure J/(kg.K)
                    'P0':1000,#Standard pressure (Hpa)
                    'R':287,#constant ....
                    'Kelvin':272.15, # constant to transform degree in kelvin
                    'E':0.622,
                    'By':'H',
                    'To':None,
                    'From':None
        }
        
        self.setpara('stanames',[])
        self.setpara('guys', {}) # ask Marcelo is it good to put the LCB object in the parameters?

    def getvarallsta(self, var = None, stanames = None, by = None, how = 'mean', From=None,To=None, From2=None, To2=None, group=None):
        """
        DESCRIPTION
            return a dataframe with the selected variable from all the stations
        TODO 
            UTLISER arg* et kwarg pour passer les argument a 
            getData sans avoir besoin de tous les recrires
        """
        if not stanames:
            stanames = self.getpara('stanames')
        
        df = pd.DataFrame()
        dfs = []
        for staname in stanames:
            print('--')
            print('Getting var of '+staname)
            station = self.getsta(staname)[0]
            s = station.getData(var = var, by = by, From=From, To=To, From2=None, To2=None, reindex=True, group=group, how=how)
            s.columns = [staname]
            
            dfs.append(s)
        dfs = pd.concat(dfs,axis=1, join='outer')
        return dfs

    def getpanel(self, stanames=None, var = None):
        """
        Get a panel constituted by the dataframe of all the stations constituing the network
        """
        if not stanames:
            stanames = self.getpara('stanames')
        
        dict_panel = {}
        for staname in stanames:
            station = self.getsta(staname)[0]
            data = station.getData(reindex=True)
            dict_panel[staname] = data
        panel = pd.Panel(dict_panel)
        return panel
        
    def getsta(self,staname, all=None, sorted=None, filter=None):
        """
        Description
            Input the name of the station and give you the station object of the network
            staname : List
            if "all==True" then return a dictionnary containing 
            all the stations names and their respective object
            
            if sorted != None. Le paramtere fourni sera utilise pour ordonner en ordre croissant la liste de stations 
            dans le reseau
            In this case it return a dictionnary with the key "staname" with the stanames sorted
            and a key stations with the coresponding stations sorted
            
            if filter != None. list of parameter Seulement les stations contenant le parametre donner seront selectionner.
        Example
            net.getsta('C04')
            net.getsta('',all=True)
            net.getsta('',all=True,sorted='lon')
            net.getsta('',all=True,sorted='lon', filter =['West'])
        """
        
        
        
        if type(staname) != list:
            try:
                if type(staname) == str:
                    staname = [staname]
            except:
                raise

        if all==True:
            try:
                staname = self.getpara('stanames')
                sta = self.getpara('guys')
            except:
                print('COULDNT GIVE THE NAME OF ALL THE STATIONS')

        else:
            try:
                sta = [ ]
                for s in staname:
                    sta.append( self.getpara('guys')[s])
            except KeyError:
                print(s)
                raise KeyError('This stations is not in the network')

        if filter != None:
            staname=Att_Sta().get_sta_id_in_metadata(filter)
            sta = [ ]
            for i in staname:
                sta.append(self.getpara('guys')[i])

        if sorted != None:
            sortednames=Att_Sta().sortsta(staname, sorted)
            sortednames=sortednames['stanames']
            sta['stanames'] =  sortednames
            s = []
            for i in sortednames:
                s.append(self.getpara('guys')[i])
            sta['stations'] = s
            print(sta)
        return sta 

    def report(self):
        guys = self.getpara('guys')
        stanames = self.getpara('stanames')
        print("Their is %d stations in the network. which are:" % len(guys))
        print(stanames)

    def dropstanan(self,var = "Ta C",by='H', perc=0, From=None,To=None):
        """
        Drop a station wich has more than <perc> percent of Nan value 
        in the the network period 
        """
        
        df = self.getvarallsta(var=var, by=by, From=From,To=To)
        sumnan = df.isnull().sum(axis=0)

        nans = (sumnan/len(df)) *100
        for sta in nans.index:
            if nans[sta] > perc:
                print("drop station "+ sta)
                self.remove(self.getsta(sta)[0])
            
    def remove(self, item):
        """
        DESCRIPTION
            Remove a station from the network
        
        """
        item._LCB_station__deregister(self)# _Com_sta - why  ? ask Marcelo
        guys = self.getpara('guys')
        stanames = self.getpara('stanames')
        del guys[item.getpara('stanames')]
        if item.getpara('stanames') in stanames: stanames.remove(item.getpara('stanames'))
        self.setpara('guys',guys)
        self.setpara('stanames', stanames)
        self.min = None
        self.max = None
        self.Data = None
#         for guy in self.getpara('guys'):
#             self.__update(guy)

    def AddFilesSta(self,files, net='ribeirao', clean=True):
        """
        Add stations to the network
        input: list of data path
        
        """
        att_var = self.att_var

        print('Adding ')
        
        if files ==[]:
            raise "The list of files is empty"



        for file in files:

            try:
                assert isinstance(file, str)

                sta = Station(file, net=net, clean=clean, att_sta=self.att_sta, att_var=att_var)
                self.add(sta)
            except AttributeError:
                print(file)
                print("Could not add station to the network")
            except AssertionError:
                print(file)
                print('file is not a string')
        print("#"*80)
        print("Network created with sucess!")
        print("#"*80)
        
    def add(self, station):
        """
            DESCRIPTION
                Add an LCB_object
        """
        self.__add_item(station)

    def validfraction(self, plot =  None):
        """
        DESCRIPTION
            Calculate the percentage of the available data in the network

        OUTPUT
            dataframe
            column "fraction" is the porcentage of available data in the network
            the other columns reprensent the availability of each stations

        EXAMPLE
            dirInPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Merge/'
            AttSta = att_sta()
            AttSta.set_filepath(dirInPath)
            Files =AttSta.getatt(AttSta.stations([]),'InPath')
        
            net=LCB_net()
            net.AddFilesSta(Files)
            
            df= net.validfraction()
            df['fraction'].resample('M').plot(kind='bar')
            plt.show()
        """
        nbsta = len(self.getpara('guys'))
        
        stations = self.getpara('guys')
        
        df = pd.DataFrame()
        for sta in stations.keys():
            data = stations[sta].Data
            data = data.sum(axis=1)
            data = data.dropna(axis=0)
            index = data.index
            s = pd.Series([1]*len(index), index = index)
            ndf = pd.DataFrame(s,columns =[sta])
            df = df.join(ndf, how = 'outer')
    
        df['fraction'] = (df.sum(axis=1)/nbsta)*100
        if plot == True:
            df['fraction'].plot()
            plt.show()
        return df
 
    def write_clean(self, outpath):
        """
        Apply threshold to the dataframe and write them 
        """ 
        threshold={
                        'Pa H':{'Min':800,'Max':1050},
                        'Ta C':{'Min':-5,'Max':40,'gradient_2min':4},#
                        'Rh %':{'Min':0.0001,'Max':100,'gradient_2min':15},
                        'Pr mm':{'Min':0,'Max':80},
                        'Sm m/s':{'Min':0,'Max':30},
                        'Dm G':{'Min':0,'Max':360},
                        'Bat mV':{'Min':0.0001,'Max':10000},
                        'Vs V':{'Min':9,'Max':9.40}#'Vs V':{'Min':8.5,'Max':9.5}
                    }

        stanames = self.getpara('stanames')
 
        
        for staname in stanames:

            station = self.getsta(staname)[0]
            filename = station.getpara('filename')
            
            newdata = station.getData(reindex=False)
#             data = station.getData(reindex=True) # old version
#             newdata = data.copy() # old version
            for var in threshold.keys():
                try:
                    index = newdata[(newdata[var]<threshold[var]['Min']) | (newdata[var]>threshold[var]['Max']) ].index
                    newdata.loc[index, var] = np.nan

                except KeyError:
                    print("No var: -> " + var)
            outfilename = outpath+filename
            
            print("o"*10)
            print("Writing clean file" +outfilename)
            newdata.to_csv(outfilename)

    def __object_name(self):
        """
        DESCRIPTION
            Create a default name for the LCB object
        """
        try:
            nbobject = len(self.getpara('stanames'))
        except:
            nbobject = 0
        return "LCB_object"+str(nbobject+1)

    def __add_item(self,item):
        
        
        try:
            staname=item.getpara('stanames')
        except KeyError:
            print("Give a default name to the LCB_object")
            staname = self.__object_name()

        print('========================================================================================')
        print('Adding To the network/group -> '+ str(staname))
        self.setpara('stanames',staname, append =True)
        self.setpara('guys',item, key= staname, append = True)
        self.getpara('guys')

        item._Station__register(self)

#     def __update(self, item):
#         """
#         DESCIPTION 
#             Calculate the mean, Min and Max of the network
#         PROBLEM
#             If their is to much data the update cause memory problems
#         """
# 
#         try:
#             print('Updating network')
#             Data=item.reindex(item.Data)
#             self.max=pd.concat((self.max,Data))
#             self.max=self.max.groupby(self.max.index).max()
#             #------------------------------------------------------------------------------ 
#             self.min=pd.concat((self.min,Data))
#             self.min=self.min.groupby(self.min.index).min()
#             #------------------------------------------------------------------------------
#             net_data=[self.Data]*(len(self.getpara('guys'))-1)# Cette technique de concatenation utilise beaucoup de mermoire
#             net_data.append(Data)
#             merged_data=pd.concat(net_data)
#             self.Data=merged_data.groupby(merged_data.index).mean()
#             self.setpara('From',self.Data.index[0])
#             self.setpara('To',self.Data.index[-1])
# 
#         except TypeError:
#             print('Initiating data network')
#             self.Data=Data
#             self.min=Data
#             self.max=Data


class Stanet_Plot():
    """
    Class container
    Contain all the plot which can be applied to a station
    """

    def __init__(self):
        pass

    def TimePlot(self,var='Ta C', by = None, group = None, subplots = None, From=None, To=None, outpath='/home/thomas/'):

            if not isinstance(var, list):
                var = [var]
            for v in var:
                data=self.getData(var=v,From=From,To=To,by=by, group = group)

                data = data.dropna()
                try:
                    data.plot(subplots = subplots, style=['o'])
                except TypeError:
                    print("No numeric data")

                objectname = self.getpara('stanames')
                if isinstance(objectname, list):
                    objectname = "network" # should be implemented somewhere else

                if outpath:
                    plt.savefig(outpath+objectname+"_TimePlot.png")
                    print('Saved at -> ' +outpath)
                    plt.close()
                else:
                    plt.show()

    def dailyplot(self,var = None,how=None, From=None, To=None,From2=None, To2=None, group= None, save= False, outpath = "/home/thomas/", labels = None):
        """
        Make a daily plot of the variable indicated
        """
        lcbplot = LCBplot() # get the plot object
#         argplot = lcbplot.getarg('plot') # get the argument by default set in the LCB plot
#         arglabel = lcbplot.getarg('label')
#         argticks = lcbplot.getarg('ticks')
#         argfig = lcbplot.getarg('figure')
#         arglegend = lcbplot.getarg('legend')

        for v in var:
            fig = plt.figure()
            color = iter(["r", "b",'g','y'])
            for from_ , to_, from2_, to2_, label in zip(From, To, From2, To2, labels):
                c = color.next()
                if how == None:
                    data = self.getData(var = v, From = from_, To=to_, From2=from2_, To2=to2_)
                    quartile1 = data.groupby(lambda x: x.hour).quantile(q=0.10)
                    quartile3 = data.groupby(lambda x: x.hour).quantile(q=0.90)
                    mean = data.groupby(lambda x: x.hour).mean()
                if how =='sum':
                    mean = self.getData(var = v,group=group, how=how, From = from_, To=to_, From2=from2_, To2=to2_)

                if how ==None:
                    print("-->" + str(quartile1.columns))

                    plt.fill_between(quartile1[v].index.values, quartile1[v].values, quartile3[v].values, alpha=0.1,color=c)

                    plt.plot([], [], color=c, alpha=0.1,linewidth=8, label=(label+' q=0.90  0.10'))


                plt.plot(mean[v].index.values, mean[v].values,linewidth = 8, linestyle='-', color=c, alpha=0.7, label=(label+' mean'))

            plt.xlim((0,24))

            if v=='Ta C':
                plt.ylabel('Temperature', fontsize=30)
            elif v=='Ua g/kg':
                plt.ylabel('Specific humidity ',fontsize=30)
            elif v=='Pr mm':
                plt.ylabel('Accumulated Precipitation',fontsize=30)
            elif var=='Ev hpa':
                plt.ylabel('Vapor Pressure',fontsize=30)
            else:
                plt.ylabel(v,fontsize=30)

            plt.xlabel( "Hours",fontsize=30)
            plt.grid(True, color="0.5")
            plt.tick_params(axis='both', which='major',labelsize=30, width=2,length=7)
            plt.legend()

            if not save:
                plt.show()
            else:
                plt.savefig(outpath+v[0:2]+"_dailyPlot.svg", transparent=True)


class LCBplot():
    def __init__(self,):
        """
        DESCRIPTION
            this class contains the methods to manage plot object
            It contains also the basin parameters that I need to make my beautiful plots
        NOTE
            This class need to be very basic to be use in all the plot of the package beeing
            developped
        
        BEFORE MODIFICATION
            LCB_object: can be either a station or a network
        This class could made to be used both on the ARPS and LCBnet program
        """

        self.argdef= {
            'OutPath':'/home/thomas/',
            'screen_width':1920,
            'screen_height':1080,
            'dpi':96,
            }
        

        self.arg = {
                 'legend':{
                           "prop":{"size":20}, # legend size
                           "loc":"best"
                           },
                 'figure':{
                        },
                 'plot':{
                         'markersize':10,
                         'alpha':1
                         },
                 'label':{
                          "fontsize":30
                         },
                'ticks':{
                         'labelsize':30,
                         'width':2, 
                         'length':7
                         
                         }
                     
                }
        self.__figwidth()

    def __figwidth(self):
        width=self.getarg('screen_width')
        height=self.getarg('screen_height')
        DPI=self.getarg('dpi')
        wfig=width/DPI #size in inches 
        hfig=height/DPI
        self.setarg('wfig',wfig)
        self.setarg('hfig',hfig)
        self.arg['figure']['figsize'] = (wfig, hfig)
        self.arg['figure']['dpi'] = DPI
        

#     def __subtitle(self,LCB_station):
#         """
#         Write the subtitle of the plot
#         """
#         sub=LCB_station.getarg('From')
#         self.setarg('subtitle', sub)

    def setarg(self,argmeter,value):
        self.arg[argmeter]=value
        print(str(argmeter)+' has been set to -> '+ str(value))

    def getarg(self,argmeter):
        try:
            return self.arg[argmeter]
        except KeyError:
            print(argmeter + ' has been not set -> Default value used ['+str(self.argdef[argmeter])+']')
            try:
                return self.argdef[argmeter]
            except KeyError:
                print(argmeter+ ' dont exist')

    def delarg(self,varname):
        try:
            del self.arg[varname]
            print('Deleted argmeter-> ',varname)
        except KeyError:
            print('This argmeter dont exist')

    def __setargdef(self,argmeter,value):
        self.argdef[argmeter]=value
        print(str(argmeter)+' has been set by [default] to -> '+ str(value))

    def __getargdef(self,argmeter):
        try:
            return self.argdef[argmeter]
        except KeyError:
                print(argmeter+ ' by [default] dont exist')

    def __levels(self,varname):
        self.argdef['nlevel']=10# number of discrete variabel level
        self.argdef['varmax']=int(self.arps.get(varname).data.max())
        self.argdef['varmin']=int(self.arps.get(varname).data.min())
        varmax=self.getarg('varmax')
        varmin=self.getarg('varmin')
        nlevel=self.getarg('nlevel')
        levels=np.linspace(varmin,varmax,nlevel)
        return levels



