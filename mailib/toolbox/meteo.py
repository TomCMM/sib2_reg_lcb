#===============================================================================
# DESCRIPTION
#    Contains function and class for meteorology 
#===============================================================================
import numpy as np



def q(Ua, Ta, Pa):
    """

    :param Ua: pd.Dataframe, Specific humidity units ?
    :param Ta: pd.DataFrame, Temeprature units?
    :param Pa: pd.Dataframe, Pressure, units:mb
    :return:
    """
    Es=0.6112*np.exp((17.67*Ta)/(Ta+243.5))*10 #hPa
    Ws=0.622*(Es/(Pa-Es))
    q=((Ua/100)*Ws)/(1-(Ua/100)*Ws)*1000
    return q

def T_dew(Tk,r):
    """
    Calcul of the Dew point temperature from the temperature in kelevin
    and the relative humidity (ratio).
    Equation from book of TSONIS 
    """
    Tdew = Tk.values / (-1.845*10**(-4) * Tk.values * np.log(r) + 1)
    return Tdew

def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def theta(T,P):
    """
    Compute Potential temperature
    T: temperture n Degree
    P: Pressure in Hectopascale
    """
    Cp = 1004. # Specific heat at constant pressure J/(kg.K)
    R = 287. #Gaz constant ....
    P0 = 1000. #Standard pressure (Hpa)
    
    theta=T*(P0/P)**(Cp/R)
    
    return theta

def br(theta1, theta2, z1,z2):
    """
    Calcul Brunt Vaissalla frequency
    """
    g=9.81
    theta_mean = (theta1 + theta2) / 2.
    br = np.sqrt( (g / theta_mean)  * ((theta2 - theta1)/ (z2 - z1)) )
    
    return br

def froude(br, U, H):
    """
    Valley depth froude number 
    """
#     print br
#     print  br.shape
#     print br * H
#     print U 
    F = U / (br* H)  
     
    return F

def Ev(q, p):
    """

    :param q: dataframe specific humidity (g.kg-1)
    :param pa: pandas dataframe Preesure (Hpa)
    :return: vapor pressure (Hpa)
    """
    e =  0.622

    w= q* 10**-3
    Ev = (w/(w+e))*p
    return Ev

def Es(Ta):
    """
    Pressure at saturation
    
    :return: 
    """
    Es=0.6112*np.exp((17.67*Ta)/(Ta+243.5))*10 #hPa

    return Es

def vpd(Ev,Es):
    """
    vapor pressure deficit
    :return:
    """

    vpd =  Es - Ev
    return vpd

def Rh_from_Td_and_T(TD, T):
    """

    :param TD: pd.Dataframe, Dew point temperature, units:Celcius
    :param T: pd.DataFrame, Temperature units:Celcius
    :return: Relative Humidity

    Values are calculated using the August-Roche-Magnus approximation.
    source: http://bmcnoldy.rsmas.miami.edu/Humidity.html
    """

    RH =  100 * (np.exp((17.625 * TD) / (243.04 + TD)) / np.exp((17.625 * T) / (243.04 + T)))
    return RH

def Td_from_Rh_and_T(RH,T):
    """

    :param RH: pd.DataFrame, Relative Humidity, units:%
    :param T: pd.Dataframe, Temperature, units C
    :return: Dew point temperature, units:C

    Values are calculated using the August-Roche-Magnus approximation.
    source: http://bmcnoldy.rsmas.miami.edu/Humidity.html
    """
    TD = 243.04 * (np.log(RH / 100) + ((17.625 * T) / (243.04 + T))) / (
                17.625 - np.log(RH / 100) - ((17.625 * T) / (243.04 + T)))
    return TD

def T_from_Rh_and_Td(TD,RH):
    """

    :param RH: pd.DataFrame, Relative Humidity, units:%
    :param TD: pd.DataFrame, Dew point temperature, units:C
    :return: T: pd.Dataframe, Temperature, units C

    Values are calculated using the August-Roche-Magnus approximation.
    source: http://bmcnoldy.rsmas.miami.edu/Humidity.html
    """
    T = 243.04 * (((17.625 * TD) / (243.04 + TD)) - np.log(RH / 100)) / (
                17.625 + np.log(RH / 100) - ((17.625 * TD) / (243.04 + TD)))
    return T