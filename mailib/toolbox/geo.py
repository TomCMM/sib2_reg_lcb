#===============================================================================
# DESCRIPTION
#     This module contain all the function and class that I use to manipulate spatial data
#===============================================================================

import numpy as np
import math
import pandas as pd
# from clima_lib.LCBnet_lib import *

def geo_idx(dd, dd_array):
    """
      search for nearest decimal degree in an array of decimal degrees and return the index.
      np.argmin returns the indices of minium value along an axis.
      so subtract dd from all values in dd_array, take absolute value and find index of minium.
     """

    geo_idx = (np.abs(dd_array - dd)).argmin()
    return geo_idx

def PolarToCartesian(norm,theta, rot=None):
    """
    Transform polar to Cartesian where 0 = North, East =90 ....
    From where the wind is blowing !!!!!
    
    ARG:
        norm: wind speed (m/s)
        theta: wind direction (degree)
        rot: Rotate the axis of an angle in degree
    RETURN:
        U,V
    """
    if not rot:
        theta = 270 - theta

        U=norm*np.cos(theta.apply(math.radians))
        V=norm*np.sin(theta.apply(math.radians))
    else:
        theta = 270 - theta + rot
        U=norm*np.cos(theta.apply(math.radians))
        V=norm*np.sin(theta.apply(math.radians))
    return U,V

def cart2pol(u, v):
    """
    Transform Cartesian to Polar where 0 = North, East =90 ....
    From where the wind is blowing !!!!!
    ARG:
        u: zonal wind (m/s)
        v: meridional wind (m/s)
        theta: wind direction (Dm G)
        rot: Rotate the axis of an angle in degree
    RETURN:
        norm: wind speed (m/s)
        theta: wind direction (degree)
    """
    rho = np.sqrt(u**2 + v**2)
    theta = np.arctan2(-v,-u)*(180/np.pi)
    theta[theta < 0 ] = theta[theta < 0 ]+360
    theta = (360 - theta) +90
    theta[(theta <= 450) & (theta >360) ] = theta[(theta <= 450) & (theta >360) ]-360
    return rho, theta

def simple_idw(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)
    # In IDW, weights are 1 / distance
    weights = 1.0 / dist
    # Make weights sum to one
    weights /= weights.sum(axis=0)
    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi


def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T
    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])
    return np.hypot(d0, d1)

def scipy_idw(x, y, z, xi, yi):
    interp = Rbf(x, y, z, function='linear', smooth=2)
    return interp(xi, yi)

def Alt_side(stations, AttSta):
    side = AttSta.getatt(stations,'side_PC4')
    alt = AttSta.getatt(stations,'Alt')
    alt_side=[]
    for a,s in zip(alt, side):
        if s=='East':
            alt_side.append(np.min(alt) - (a-np.min(alt)))
        else:
            alt_side.append(a)


    return np.array(alt_side)

def side_PC4(stations, AttSta):
    side = pd.Series(AttSta.getatt(stations,'side'),index=stations,name="side_PC4")
    side['C04']="East" 
    return side

def is_point_in_domain(domain_lat, domain_lon, points_lat, points_lon):
    """
    PARAMETERS:
        Latitude
        Longitude
    RETURN
        a list of true or fal
    """
    latmodelmin = domain_lat.min()
    latmodelmax = domain_lat.max()
    lonmodelmin = domain_lon.min()
    lonmodelmax = domain_lon.max()
    points_lat = pd.Series(points_lat, name='Lat')
    points_lon = pd.Series(points_lon, name='Lon')
    points_lat = points_lat[(points_lat > latmodelmin) & (points_lat < latmodelmax) ]
    points_lon = points_lon[(points_lon > lonmodelmin) & (points_lon < lonmodelmax) ]
    point_lat_lons = pd.concat([points_lat, points_lon], axis=1, join='outer')

    point_lat_lons.dropna(inplace=True)

    try:
        return point_lat_lons['Lat'], point_lat_lons['Lon'] # replace by array
    except KeyError:
        return pd.DataFrame(columns=['lat']), pd.DataFrame(columns=['Lon'])



def get_gridpoint_position(model_lat, model_lon, points_lat, points_lon, is_in_domain=True):
    """
    Parameters:
        lat: A serie with the latitude of the point to select
        lon: A serie with the longitude of the point to select
    Return:
        A dataframe with the i,j position in the grid
    """

    if is_in_domain:
        points_lat, points_lon =  is_point_in_domain(model_lat, model_lon, points_lat, points_lon)
        index = points_lat.index
        points_lat = points_lat.values
        points_lon = points_lon.values

    Is = []
    Js = []

    for point_lat, point_lon in zip(points_lat, points_lon):
        distance_i = (model_lat - point_lat)**2 
        distance_j = (model_lon - point_lon)**2
        I = np.where(distance_i == distance_i.min())
        J = np.where(distance_j == distance_j.min())
        Is.append(I[0][0])
        Js.append(J[0][0])
    print(Is)
    print(Js)

    idxs = pd.DataFrame([Is, Js])
    idxs = idxs.T

    if is_in_domain:
        idxs.index = index
        idxs.columns = ['i','j']
    return idxs




