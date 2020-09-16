#!/usr/bin/python
"""programme to make quantile(CDF)-mapping bias correction"""
"""Adrian Tompkins tompkins@ictp.it - please feel free to use"""

import numpy as np



#----------------
# MAIN CODE START
#----------------
# global cdfobs,cdfmod,xbins
n=100
cdfn=10
# make some fake observations(obs) and model (mod) data:
obs=np.random.uniform(low=0.5, high=13.3, size=(n,))
mod=np.random.uniform(low=1.4, high=19.3, size=(n,))



class qm(object):
    """


    obs: observation
    mod: model
    cdfn: number of class
    """
    def __init__(self,obs,mod,cdfn):
        self.obs = obs
        self.mod = mod
        self.cdfn = cdfn

    def qm_mapping(self):
        """
        Function fodund in

        :param obs:
        :param mod:
        :return:
        """


        # sort the arrays
        obs = np.sort(self.obs)
        mod = np.sort(self.mod)

        # calculate the global max and bins.
        global_max=max(np.amax(obs),np.amax(mod))
        global_min=min(np.amin(obs),np.amin(mod))


        wide=(global_max - global_min)/self.cdfn


        xbins=np.arange(global_min, global_max+wide, wide)

        # create PDF
        pdfobs,bins=np.histogram(obs,bins=xbins)
        pdfmod,bins=np.histogram(mod,bins=xbins)

        # create CDF with zero in first entry.
        cdfobs = np.insert(np.cumsum(pdfobs),0,0.0)
        cdfmod = np.insert(np.cumsum(pdfmod),0,0.0)

        return cdfobs, cdfmod, xbins

    def map(self, vals):
        """ CDF mapping for bias correction """
        """ note that values exceeding the range of the training set"""
        """ are set to -999 at the moment - possibly could leave unchanged?"""
        # calculate exact CDF values using linear interpolation

        cdfobs, cdfmod, xbins = self.qm_mapping()

        cdf1 = np.interp(vals, xbins, cdfmod, left=0.0, right=999.0)
        # now use interpol again to invert the obsCDF, hence reversed x,y
        corrected = np.interp(cdf1, cdfobs, xbins, left=0.0, right=-999.0)
        return corrected



if __name__ == '__main__':
    # cdfobs, cdfmod, xbins = qm_mapping(obs, mod)


    # dummy model data list to be bias corrected
    raindata=[2.0,5.0]
    print(raindata)
    print(map(raindata, xbins, cdfobs, cdfmod))

