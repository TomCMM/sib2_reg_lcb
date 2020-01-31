"""
    Description:
        Plot the metadata altitude in function of the DEM altitude from the metadata latitude and longitude

    Parameters
        metadata_path: string, path of the metadata file from which take the DEM altitude
        savefig_path: path and name of the check figure
"""


# Check the altitude in the metadata
from stanet.lcbstanet import *
import matplotlib.pyplot as plt

def plot_check_altitude(metadata_path, savefig_path, sta_ids):
    AttSta = Att_Sta(metadata_path)
    fig, ax = plt.subplots()
    plt.scatter(AttSta.attributes.loc[sta_ids,'alt'], AttSta.attributes.loc[sta_ids,'alt_dem'])
    for i, txt in enumerate(AttSta.attributes.index):
        ax.annotate(txt, (AttSta.attributes.loc[txt,'alt'], AttSta.attributes.loc[txt,'alt_dem']))
    # plt.show()
    plt.savefig(savefig_path)


if __name__ == '__main__':
    # User input
    metadata_path = '../../1_database/metadata/topo_index_att/df_topoindex_regional.csv'
    savefig_path = '../res/check_altitude.png'

    plot_check_altitude(metadata_path, savefig_path)

