
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
# Read scores
in_scores = "/home/thomas/phd/framework/visu/data/daily_obs_scores/"

PC1T_scores = pd.read_csv(in_scores+ 'PC1T.csv', index_col=0, header=None)
PC2T_scores = pd.read_csv(in_scores+ 'PC2T.csv', index_col=0, header=None)
PC3T_scores = pd.read_csv(in_scores+ 'PC3T.csv', index_col=0, header=None)
PC4T_scores = pd.read_csv(in_scores+ 'PC4T.csv', index_col=0, header=None)

PC1U_scores = pd.read_csv(in_scores+ 'PC1U.csv', index_col=0, header=None)
PC2U_scores = pd.read_csv(in_scores+ 'PC2U.csv', index_col=0, header=None)

PC1V_scores = pd.read_csv(in_scores+ 'PC1V.csv', index_col=0, header=None)
PC2V_scores = pd.read_csv(in_scores+ 'PC2V.csv', index_col=0, header=None)
PC3V_scores = pd.read_csv(in_scores+ 'PC3V.csv', index_col=0, header=None)

# Get images animation

in_anim = "/home/thomas/phd/framework/visu/res/pres/"
out_anim= "/home/thomas/phd/framework/visu/res/pres_with_scores/"

# Full



def anim_full(PC_var, label_name,slidename, folder, showcolorbar=True, vmin=None, vmax=None, full=True):

    print('-------------')
    hours = range(23)
    print(label_name)

    for hour in hours:

        print(str(hour))

        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(15, 10)
        gs.update(wspace=0, hspace=0)


        # ax1 = plt.subplot(gs[:3, 3:7])
        # ax2 = plt.subplot(gs[1, 1:4])


        ax3 = plt.subplot(gs[:, :])


        ax2 = fig.add_axes([0.3, 0.05, 0.4, 0.025])



        # plot colorbar
        if showcolorbar:
            cmap = matplotlib.cm.coolwarm
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cb = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,
                                            norm=norm,
                                            orientation='horizontal')
            cb.set_label(label_name)



        # Plot animation image
        img=mpimg.imread(in_anim+folder+str(hour).zfill(2)+".png")
        imgplot = ax3.imshow(img)
        ax3.axis('off')

        # Annotate hours
        ax3.annotate('Hours:  ' +str(hour).zfill(2)+'h',xy=(.45, 0.87), xycoords='figure fraction', fontsize=26)

        # slide name
        bbox_props = dict(boxstyle="square,pad=0.5", fc="0.9", ec="0.7", lw=2)
        ax3.annotate(slidename,xy=(.2, 0.95), xycoords='figure fraction', fontsize=22, bbox=bbox_props)


        # Rectanlge
        # matplotlib.patches.Rectangle(xy=(.1, 0.95), width=10, height=50, color='k')

        plt.savefig(out_anim+folder+str(hour).zfill(2)+".png", dpi=150)
        plt.close('all')


def anim_PC(PC_var, label_name,slidename, folder,showcolorbar=True, vmin=None, vmax=None, outpath=None):

    print('-------------')
    hours = range(23)
    print(label_name)

    for hour in hours:

        print(str(hour))

        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(15, 10)
        gs.update(wspace=0, hspace=0)


        ax1 = plt.subplot(gs[:3, 3:7])
        # ax2 = plt.subplot(gs[1, 1:4])


        ax3 = plt.subplot(gs[4:, :])

        if showcolorbar:
            ax2 = fig.add_axes([0.3, 0.05, 0.4, 0.025])

        # plot scores
        PC_var.plot(ax=ax1, legend=None, color='k', linewidth=4)
        # ax1.xaxis.set_visible(False)
        ax1.xaxis.label.set_visible(False)
        ax1.set_ylabel('PC scores')
        ax1.grid(True)
        # ax1.axhline(0, linewidth=2, c='0.5')
        ax1.axvline(hour, linewidth=2, c='0.5')
        ax1.scatter(hour,PC_var.iloc[hour], s=80,c='0.5', alpha=0.5)


        # plot colorbar
        if showcolorbar:
            cmap = matplotlib.cm.coolwarm
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cb = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,
                                            norm=norm,
                                            orientation='horizontal')
            cb.set_label(label_name)



        # Plot animation image
        img=mpimg.imread(in_anim+folder+str(hour).zfill(2)+".png")
        imgplot = ax3.imshow(img)
        ax3.axis('off')

        # Annotate hours
        ax1.annotate('Hours:  ' +str(hour).zfill(2)+'h',xy=(.45, 0.70), xycoords='figure fraction', fontsize=18)

        # slide name
        bbox_props = dict(boxstyle="square,pad=0.5", fc="0.9", ec="0.7", lw=2)
        ax1.annotate(slidename,xy=(.3, 0.95), xycoords='figure fraction', fontsize=22, bbox=bbox_props)


        # Rectanlge
        # matplotlib.patches.Rectangle(xy=(.1, 0.95), width=10, height=50, color='k')

        print(out_anim+outpath+str(hour).zfill(2)+".png")
        plt.savefig(out_anim+outpath+str(hour).zfill(2)+".png", dpi=150)
        plt.close('all')


anim_full(PC1T_scores, "full","Absolute temperature and wind reconstructed - daily cycle" ,"FULL/", showcolorbar=True, vmin=14,vmax=28)


# anim_PC(PC1T_scores, "Temperature (C)","PC1 Temperature reconstructed - Daily cycle" ,"T/PC1/",outpath = "T/PC1/", showcolorbar=True, vmin=14,vmax=28)
# anim_PC(PC2T_scores, "Temperature (C)", "PC2 Temperature reconstructed - Daily cycle" ,"T/PC2/",outpath = "T/PC3/", showcolorbar=True, vmin=-2, vmax=1)
# anim_PC(PC4T_scores, "Temperature (C)","PC4 Temperature reconstructed - Daily cycle" , "T/PC3/",outpath = "T/PC4/", showcolorbar=True, vmin=-0.5, vmax=0.5)
# anim_PC(PC3T_scores, "Temperature (C)","PC3 Temperature reconstructed - Daily cycle" , "T/PC4/",outpath = "T/PC2/", showcolorbar=True, vmin=-3, vmax=2)


anim_PC(PC1T_scores, "Temperature (C)","PC1 Temperature reconstructed - Daily cycle" ,"T/PC1/",outpath = "T/PC1/", showcolorbar=True, vmin=14,vmax=28)
anim_PC(PC2T_scores, "Temperature (C)", "PC2 Temperature reconstructed - Daily cycle" ,"T/PC4/",outpath = "T/PC2/", showcolorbar=True, vmin=-2, vmax=1)
anim_PC(PC3T_scores, "Temperature (C)","PC3 Temperature reconstructed - Daily cycle" , "T/PC2/",outpath = "T/PC3/", showcolorbar=True, vmin=-3, vmax=2)
anim_PC(PC4T_scores, "Temperature (C)","PC4 Temperature reconstructed - Daily cycle" , "T/PC3/",outpath = "T/PC4/", showcolorbar=True, vmin=-0.5, vmax=0.5)

#
#
# anim_PC(PC2U_scores, "a","PC2 Zonal wind reconstructed - Daily cycle" ,"U/PC1/",outpath = "U/PC2/", showcolorbar=False, vmin=14,vmax=28)
# anim_PC(PC1U_scores, "a", "PC1 Zonal wind reconstructed - Daily cycle" ,"U/PC2/",outpath = "U/PC1/", showcolorbar=False, vmin=-2, vmax=1)
#
# anim_PC(PC3V_scores, "a","PC3 Meridional wind reconstructed - Daily cycle" , "V/PC1/",outpath = "V/PC3/", showcolorbar=False, vmin=-0.5, vmax=0.5)
# anim_PC(PC1V_scores, "a","PC1 Meridional wind reconstructed - Daily cycle" , "V/PC2/",outpath = "V/PC1/", showcolorbar=False, vmin=-0.5, vmax=0.5)
# anim_PC(PC2V_scores, "a","PC2 Meridional wind reconstructed - Daily cycle" , "V/PC3/",outpath = "V/PC2/", showcolorbar=False, vmin=-0.5, vmax=0.5)
