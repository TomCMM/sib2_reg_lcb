#===============================================================================
# DESCRIPTION
#     PLot the variables in function of the altitude
#===============================================================================

import glob
import LCBnet_lib
from LCBnet_lib import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import statsmodels.api as sm
import matplotlib
from scipy.interpolate import interp1d
from collections import Counter
from scipy import interpolate
from LapseRate import AltitudeAnalysis

 
if __name__=='__main__':
#     #===========================================================================
#     #  Get INNMET Files
#     #===========================================================================
#     Path='/home/thomas/INMET/'
#     #     Path='/home/thomas/MergeDataThreeshold/'
#     OutPath='/home/thomas/INMET_timeserie/'
#     Files=glob.glob(Path+"*")
# 
#     altanal = AltitudeAnalysis(Files, net='INMET')
# 
#     #===========================================================================
#     # Plot var in function of Altitude
#     #===========================================================================
# 
#     hours = np.arange(15,24,1)
#     hours = [15,9]
#     altanal.VarVsAlt(vars= ['Ta C'], by= 'H',  dates = hours, From='2013-01-01 00:00:00', To='2016-01-01 00:00:00')
#     altanal.plot(analysis = 'var_vs_alt', marker_side = True, annotate = True, print_= True)
#     
    #===========================================================================
    #  Get IAC input Files
    #===========================================================================
#     Path='/home/thomas/IAC/'
#     #     Path='/home/thomas/MergeDataThreeshold/'
#     OutPath='/home/thomas/IAC_timeserie/'
#     Files=glob.glob(Path+"*")
# 
#     altanal = AltitudeAnalysis(Files, net='IAC')
#     
    #===========================================================================
    # Get Sinda input Files
    #===========================================================================

#     Path = '/home/thomas/Sinda/'
#     OutPath='/home/thomas/IAC_timeserie/'
#     
#     Files=glob.glob(Path+"*")
#     altanal = AltitudeAnalysis(Files, net='Sinda')
# 
#     #===========================================================================
#     # Plot var in function of Altitude
#     #===========================================================================
# 
# #     hours = np.arange(15,24,1)
#     hours = [12,6]
#     altanal.VarVsAlt(vars= ['Ta C'], by= 'H',  dates = hours, From='2013-01-01 00:00:00', To='2016-01-01 00:00:00')
#     altanal.plot(analysis = 'var_vs_alt', marker_side = True, annotate = True, print_= True)