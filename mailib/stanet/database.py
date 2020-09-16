"""
    DESCRIPTION
        Module to create stations database and extract data from it

    TODO This could be improved with a SQL database with e.g. SQLalchemy module
"""

import os
import pandas as pd
from mailib.stanet.lcbstanet import Att_Sta, Att_Var, Net

class StaNetDatabase():
    """


    """

    def __init__(self, path_database, relative_path_data, relative_path_metadata, metadatafile):
        print('Initialise stanet database')
        self.path_database = path_database
        self.path_data_folder = path_database + relative_path_data
        self.path_metadata_folder = path_database + relative_path_metadata
        self.path_metadatafile = self.path_metadata_folder + metadatafile


    def get_var_nets(self, network_folder_names=None, var=None,
                     params_selection=None, by=None, From=None, To=None, **kwargs):
        """

        :param networks_folder: list of networks folder names with
        the same name than the folder found in data (see database path_data_folder)
                params selections: dictionnary with attribute to select in between
                    ex: {'lat':Lat, 'lon':Lon}
        :return:
            Pandas Dataframe object, columns


        Example:
            df_daily = stanet_database.get_all_net(daily_network_folder_name,
             var, metadatafile, params_selection, by='D', From=None, To = None) # Create the dataframe

        """

        inpath_data = self.path_data_folder

        dfs = []
        for netname in network_folder_names:
            print('netname')
            df = self.__get_dataframe_net(var, netname, inpath_data, params_selection, by=by, From=From, To=To, **kwargs)
            dfs.append(df)

        dfs = pd.concat(dfs, axis=1, join='outer')

        return dfs

    def __get_dataframe_net(self, var, netname, inpath_data, params_selection,by=None, **kwargs):
        """
        Retrun dataframe for one network

        :return:
            pandas dataframe, with the selected stations, network, variables
        """

        att_net = Att_Sta(self.path_metadatafile)


        sta_id = att_net.get_sta_id_in_metadata(values=[netname], params=params_selection)  # get the selected id in the network
        att_net.set_filepath(stanames=sta_id,
                             file_path=inpath_data + netname + '/')  # set the path of the data to the attribute

        files = att_net.getatt(sta_id, 'InPath')  # get list path of data

        att_var = Att_Var()
        net = Net(att_net, att_var=att_var)  # Create instance of network
        net.AddFilesSta(files, net=netname)  # File the object network with the data

        df = net.getvarallsta(var=var, by=by, From=None, To=None)  # get variables and retrun dataframe
        return df


def read_database(path_database, metadatafile = 'database_metadata.csv'):
    """
    Read a stations networks database

    Following structure

    database
        data
            net1
            net2
                sta1,csv
                sta2

        raw_data
            net1
            net2
                sta1
                sta2
        metadata
            all_stations_metadata

    :return:stanet database object

    """

    relative_path_data = 'data/'
    relative_path_metadata = 'metadata/'

    print(f'Check database structure: {path_database}')


    #TODO Make a strcit lecture of the database
    assert os.path.exists(path_database) , str(path_database)
    assert os.path.exists(path_database + relative_path_data) , str(path_database +relative_path_data)
    assert os.path.exists(path_database + relative_path_metadata), str(path_database + relative_path_metadata)



    if not os.path.exists(path_database + relative_path_metadata + metadatafile):
        raise(f'The metadata file: <{metadatafile}> is not found in {path_database}{relative_path_metadata}')

    database = StaNetDatabase(path_database, relative_path_data, relative_path_metadata, metadatafile)
    return database



