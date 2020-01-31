# Check the altitude in the metadata
from stadata.lib.LCBnet_lib import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    AttSta = att_sta('/home/thomas/phd/framework/predictors/out/att_sta/df_topoindex_regional.csv')


    fig, ax = plt.subplots()
    plt.scatter(AttSta.attributes.loc[:,'Alt'], AttSta.attributes.loc[:,'alt_dem'])
    for i, txt in enumerate(AttSta.attributes.index):
        ax.annotate(txt, (AttSta.attributes.loc[txt,'Alt'], AttSta.attributes.loc[txt,'alt_dem']))

    plt.show()
