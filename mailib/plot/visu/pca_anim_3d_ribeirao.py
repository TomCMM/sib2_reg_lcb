"""
Description:
    Visualisation of the Ribeirao Das Posses meteorological variability in 3D

Doc
http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#mayavi.mlab.quiver3d
http://gael-varoquaux.info/programming/mayavi-representing-an-additional-scalar-on-surfaces.html
http://hplgit.github.io/primer.html/doc/pub/plot/._plot-readable007.html
https://github.com/yxdragon/myvi
http://www.scipy-lectures.org/advanced/3d_plotting/
"""

from mayavi.mlab import *
from mayavi import mlab
import numpy as np
from local_downscaling import concat_dfs
import pandas as pd
import matplotlib

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

def rotat_visu():
    hours = np.tile(np.repeat(range(0,24),5),3)
    azimuths = np.arange(0,360)
    distances = np.append(np.linspace(20000, 6000, 120),np.repeat(6000, 240))
    for azimuth, hour, distance in zip(azimuths[234:], hours[234:], distances[234:]):
        print(hour)
        T = concat_dfs(inpath, column = hour, regex="TPC")
        Q = concat_dfs(inpath, column = hour, regex="QPC")
        U = concat_dfs(inpath, column = hour, regex="UPC")
        V = concat_dfs(inpath, column = hour, regex="VPC")

        # Sum the PCs
        T = T.sum(axis=1)
        Q = Q.sum(axis=1)
        U = U.sum(axis=1)
        V = V.sum(axis=1)

        T = T.values.reshape(239,244)
        Q = Q.values.reshape(239,244)
        U = U.values.reshape(239,244)
        V = V.values.reshape(239,244)
        W = np.zeros(U.shape)
        s = np.ones(U.shape)

        mlab.figure(bgcolor=(0.7,0.7,0.7), fgcolor=(0.,0.,0.))

        # # Surface plot
        pts = mlab.quiver3d(x, y, z,s,s,s, scalars=T,  mode='cube', scale_factor=0, colormap='bwr')
        pts.glyph.color_mode = "color_by_scalar"
        pts.glyph.glyph_source.glyph_source.center = [0,0,0]

        # pts.module_manager.scalar_lut_manager.lut.table = T
        # draw()

        mesh = mlab.pipeline.delaunay2d(pts)     # Create and visualize the mesh
        surf = mlab.pipeline.surface(mesh, colormap='bwr', vmin=12, vmax=26)
        # surf.module_manager.scalar_lut_manager.reverse_lut = True # reverse colormap
        sc = mlab.scalarbar(surf, title="Temperature (C)", label_fmt="%.0f")

        sc.scalar_bar_representation.position = [0.1, 0.9]
        sc.scalar_bar_representation.position2 = [0.8, 0.05]


        # Quiver plot
        mask_point=4
        quiver3d(x[::mask_point,::mask_point],y[::mask_point,::mask_point],z[::mask_point,::mask_point]+50,
                 U[::mask_point,::mask_point], V[::mask_point,::mask_point], W[::mask_point,::mask_point],
                 opacity=0.8,
                 mode='2darrow', color=(0.3,0.3,0.3), scale_factor=150, transparent=True)

        # mlab.axes()

        mlab.text(0.45, 0.97, "Time: "+str(hour).zfill(3) + "H", width=0.1, line_width=0.01, color=(0,0,0))


        mlab.view(azimuth=azimuth, elevation=60, distance=distance, focalpoint=np.array([3500,2500,0]))
        mlab.savefig(outpath+str(azimuth).zfill(3)+".png",size=(1420,1080), magnification='auto')
        mlab.close()
        # show()


def full():
    # Get reconstructed variables field
    inpath = "/home/thomas/phd/framework/model/out/local_downscaling/ribeirao_articleII/pres_obs_3d/"
    outpath = "/home/thomas/phd/framework/visu/res/pres/FULL/"


    # 360 numbers between 0 and 24h
    hours = range(0,23)
    for hour in hours:

        print(hour)
        T = concat_dfs(inpath, column = hour, regex="TPC")
        Q = concat_dfs(inpath, column = hour, regex="QPC")
        U = concat_dfs(inpath, column = hour, regex="UPC")
        V = concat_dfs(inpath, column = hour, regex="VPC")

        # Sum the PCs
        T = T.sum(axis=1)
        # Q = Q.sum(axis=1)
        U = U.sum(axis=1)
        V = V.sum(axis=1)

        T = T.values.reshape(239,244)
        # Q = Q.values.reshape(239,244)
        U = U.values.reshape(239,244)
        V = V.values.reshape(239,244)
        W = np.zeros(U.shape)
        s = np.ones(U.shape)

        mlab.figure(bgcolor=(0.7,0.7,0.7), fgcolor=(0.,0.,0.))

        # # Surface plot
        pts = mlab.quiver3d(x, y, z,s,s,s, scalars=T,  mode='cube', scale_factor=0, colormap='bwr')
        pts.glyph.color_mode = "color_by_scalar"
        pts.glyph.glyph_source.glyph_source.center = [0,0,0]

        # pts.module_manager.scalar_lut_manager.lut.table = T
        # draw()

        mesh = mlab.pipeline.delaunay2d(pts)     # Create and visualize the mesh
        surf = mlab.pipeline.surface(mesh, colormap='coolwarm', vmin=12, vmax=26)
        # surf.module_manager.scalar_lut_manager.reverse_lut = True # reverse colormap
        # sc = mlab.scalarbar(surf, title="Temperature (C)", label_fmt="%.0f")
        #
        # sc.scalar_bar_representation.position = [0.2, 0.85]
        # sc.scalar_bar_representation.position2 = [0.6, 0.08]

        # Quiver plot
        mask_point=4
        quiver3d(x[::mask_point,::mask_point],y[::mask_point,::mask_point],z[::mask_point,::mask_point]+100,
                 U[::mask_point,::mask_point], V[::mask_point,::mask_point], W[::mask_point,::mask_point],
                 opacity=0.8,
                 mode='2darrow', color=(0.3,0.3,0.3), scale_factor=150, transparent=True)

        # mlab.axes()

        # mlab.text(0.45, 0.97, "Time: "+str(hour).zfill(2) + "H", width=0.1, line_width=0.01, color=(0,0,0))


        mlab.view(azimuth=255, elevation=40, distance=6500, focalpoint=np.array([3500,2600,0]))
        mlab.savefig(outpath+str(hour).zfill(2)+".png",size=(1420,1080), magnification='auto')
        mlab.close()


def T_PCs(nbpc=0, vmin=None, vmax=None):
    # Get reconstructed variables field
    inpath = "/home/thomas/phd/framework/model/out/local_downscaling/ribeirao_articleII/pres_obs_3d/"
    outpath = "/home/thomas/phd/framework/visu/res/pres/T/PC"+str(nbpc+1)+"/"


    # 360 numbers between 0 and 24h
    hours = range(0,23)
    for hour in hours:

        print(hour)
        T = concat_dfs(inpath, column = hour, regex="TPC")
        # Q = concat_dfs(inpath, column = hour, regex="QPC")
        # U = concat_dfs(inpath, column = hour, regex="UPC")
        # V = concat_dfs(inpath, column = hour, regex="VPC")

        # Sum the PCs
        T = T.iloc[:,nbpc]
        # Q = Q.sum(axis=1)
        # U = U.sum(axis=1)
        # V = V.sum(axis=1)

        T = T.values.reshape(239,244)
        # Q = Q.values.reshape(239,244)
        # U = U.values.reshape(239,244)
        # V = V.values.reshape(239,244)
        # W = np.zeros(U.shape)
        s = np.ones((239,244))

        mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.))

        # # Surface plot
        pts = mlab.quiver3d(x, y, z,s,s,s, scalars=T,  mode='cube', scale_factor=0, colormap='bwr')
        pts.glyph.color_mode = "color_by_scalar"
        pts.glyph.glyph_source.glyph_source.center = [0,0,0]

        # pts.module_manager.scalar_lut_manager.lut.table = T
        # draw()

        mesh = mlab.pipeline.delaunay2d(pts)     # Create and visualize the mesh
        surf = mlab.pipeline.surface(mesh, colormap='coolwarm', vmin=vmin, vmax=vmax)
        # surf.module_manager.scalar_lut_manager.reverse_lut = True # reverse colormap
        # sc = mlab.scalarbar(surf, title="Temperature (C)", label_fmt="%.1f")

        # sc.scalar_bar_representation.position = [0.2, 0.85]
        # sc.scalar_bar_representation.position2 = [0.6, 0.08]

        # sc.scalar_bar_representation.position = [0.2, 0]
        # sc.scalar_bar_representation.position2 = [0.6, 0.08]
        #
        #
        # # Quiver plot
        # mask_point=4
        # quiver3d(x[::mask_point,::mask_point],y[::mask_point,::mask_point],z[::mask_point,::mask_point]+100,
        #          U[::mask_point,::mask_point], V[::mask_point,::mask_point], W[::mask_point,::mask_point],
        #          opacity=0.8,
        #          mode='2darrow', color=(0.3,0.3,0.3), scale_factor=150, transparent=True)

        # mlab.axes()

        # mlab.text(0.78, 0.13, "Time: "+str(hour).zfill(2) + "H", width=0.2, line_width=0.01, color=(0,0,0))


        mlab.view(azimuth=255, elevation=40, distance=6500, focalpoint=np.array([3500,2600,0]))
        # mlab.view(azimuth=260, elevation=45, distance=7000, focalpoint=np.array([3250,2000,0]))

        mlab.savefig(outpath+str(hour).zfill(2)+".png",size=(1420,1080), magnification='auto')
        mlab.close()
        # show()


def UV_PCs(nbpc=0, use_U=False, scale_factor=150):
    # Get reconstructed variables field
    inpath = "/home/thomas/phd/framework/model/out/local_downscaling/ribeirao_articleII/pres_obs_3d/"

    if use_U:
        outpath = "/home/thomas/phd/framework/visu/res/pres/U/"+"PC"+str(nbpc+1)+"/"
    else:
        outpath = "/home/thomas/phd/framework/visu/res/pres/V/"+"PC"+str(nbpc+1)+"/"


    # 360 numbers between 0 and 24h
    hours = range(0,23)
    for hour in hours:

        print(hour)
        # T = concat_dfs(inpath, column = hour, regex="TPC")
        # Q = concat_dfs(inpath, column = hour, regex="QPC")
        U = concat_dfs(inpath, column = hour, regex="UPC")
        V = concat_dfs(inpath, column = hour, regex="VPC")

        # Sum the PCs
        # T = T.iloc[:,nbpc]
        # Q = Q.sum(axis=1)


        if use_U == False:
            U = U.iloc[:,0]
            V = V.iloc[:,nbpc]
        else:
            U = U.iloc[:,nbpc]
            V = V.iloc[:,0]

        # T = T.values.reshape(239,244)
        # Q = Q.values.reshape(239,244)
        U = U.values.reshape(239,244)
        V = V.values.reshape(239,244)

        if use_U == False:
            U = np.zeros(U.shape)
        else:
            V = np.zeros(V.shape)

        W = np.zeros(U.shape)
        s = np.ones(U.shape)
        T = np.ones(U.shape)

        mlab.figure(bgcolor=(0.7,0.7,0.7), fgcolor=(0.,0.,0.))



        # # # Surface plot
        pts = mlab.quiver3d(x, y, z,s,s,s, scalars=T,  mode='cube', scale_factor=0, colormap='bwr')
        pts.glyph.color_mode = "color_by_scalar"
        pts.glyph.glyph_source.glyph_source.center = [0,0,0]

        pts.module_manager.scalar_lut_manager.lut.table = T
        draw()

        mesh = mlab.pipeline.delaunay2d(pts)     # Create and visualize the mesh
        surf = mlab.pipeline.surface(mesh, colormap='Greys', vmin=-3, vmax=10)
        # surf.module_manager.scalar_lut_manager.reverse_lut = True # reverse colormap
        # sc = mlab.scalarbar(surf, title="Temperature (C)", label_fmt="%.0f")
        #
        # sc.scalar_bar_representation.position = [0.1, 0.9]
        # sc.scalar_bar_representation.position2 = [0.8, 0.05]
        #
        #
        # Quiver plot
        mask_point=4
        quiver3d(x[::mask_point,::mask_point],y[::mask_point,::mask_point],z[::mask_point,::mask_point]+100,
                 U[::mask_point,::mask_point], V[::mask_point,::mask_point], W[::mask_point,::mask_point],
                 opacity=0.8,line_width=3,
                 mode='2darrow', color=(0.1,0.1,0.1), scale_factor=scale_factor, transparent=True)
        #
        # mlab.axes()
        #
        # mlab.text(0.45, 0.97, "Time: "+str(hour).zfill(2) + "H", width=0.1, line_width=0.01, color=(0,0,0))


        mlab.view(azimuth=255, elevation=40, distance=6500, focalpoint=np.array([3500,2600,0]))
        mlab.savefig(outpath+str(hour).zfill(2)+".png",size=(1420,1080), magnification='auto')
        mlab.close()
        # show()

if __name__ == '__main__':

    # Get posittion of the output model field
    df_field_topex_alt = pd.read_csv('/home/thomas/phd/framework/predictors/out/df_field/ribeirao/df_field_ribeirao.csv', index_col=0)
    z = df_field_topex_alt.loc[:,'Alt'].reshape(239,244)
    y = df_field_topex_alt.loc[:,'lat'].reshape(239,244)
    x = df_field_topex_alt.loc[:,'lon'].reshape(239,244)

    # Rescale
    x = x* 110000 # in meters
    y = y* 110000 # in meters
    x = x-x.min()
    y = y-y.min()
    z = z-z.min()
    z = z*2 # increase topographical aspect

    # Plot the graphics
    # full()
    # T_PCs(nbpc=0, vmin=14,vmax=28)
    # T_PCs(nbpc=1, vmin=-2, vmax=1)
    # T_PCs(nbpc=2, vmin=-0.5, vmax=0.5)
    # T_PCs(nbpc=3, vmin=-3, vmax=2)
    # #
    UV_PCs(nbpc=0, use_U=True, scale_factor=300)
    UV_PCs(nbpc=1, use_U=True, scale_factor=150)

    UV_PCs(nbpc=0, use_U=False, scale_factor=200)
    UV_PCs(nbpc=1, use_U=False, scale_factor=150)
    UV_PCs(nbpc=2, use_U=False, scale_factor=150)
    # #



    print('done')




###############################################
   #
    # lat = raster_lat[:,0]
    # lon = raster_lon[:0,:]
    #
    #
    #
    # mask_posses_array = df_field_topex_alt.loc[:,'mask_posses'].reshape(239,244)
    # mask_posses_array = 1 - mask_posses_array
    #
    # raster_val = np.ma.array(raster_val, mask= mask_posses_array)

# x= x.T
# y = y.T

# xmin = x.min()
# xmax = x.max()
#
# ymin =x.min()
# ymax = x.max()
#
# zmin = x.min()
# zmax = x.max()
#
#
# # Plot surface
# mlab.surf(x,y,z,extent=[xmin, xmax, ymin, ymax, zmin,zmax], colormap='terrain', warp_scale='auto')
# mlab.show()

# # Surface with values as colors
# # Create the data source
# src = mlab.pipeline.array2d_source(z)
#
# dataset = src.mlab_source.dataset
# array_id = dataset.point_data.add_array(w.T.ravel())
# dataset.point_data.get_array(array_id).name = 'color'
# dataset.point_data.update()
#
# # Here, we build the very exact pipeline of surf, but add a
# # set_active_attribute filter to switch the color, this is code very
# # similar to the code introduced in:
# # http://code.enthought.com/projects/mayavi/docs/development/html/mayavi/mlab.html#assembling-pipelines-with-mlab
# warp = mlab.pipeline.warp_scalar(src, warp_scale=.1)
# normals = mlab.pipeline.poly_data_normals(warp)
# active_attr = mlab.pipeline.set_active_attribute(normals,
#                                             point_scalars='color')
# surf = mlab.pipeline.surface(active_attr)
#
# # # Finally, add a few decorations.
# mlab.axes()
# # mlab.outline()
# # mlab.view(-177, 82)
# mlab.show()

