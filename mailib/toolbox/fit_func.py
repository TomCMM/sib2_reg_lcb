#===============================================================================
# DESCRIPTION
#    Function for curvit 
#===============================================================================
import numpy as np


def piecewise_linear_pc3_ribeirao(x, k1, k2, x0, y0): # need to be improve!!!!!!!!!!!!!!!

    x0 = 1100
    y0 = -0.3
    try:
        index = x.index
        x = x.values
    except AttributeError:
        pass
    return np.piecewise(x, [x < x0, x > x0], [lambda x: k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

# def piecewise_linear(x,a1,b1,a2,b2):
#     return np.piecewise(x, [x<1100], [lambda x: a1*x + b1, lambda x: a2*x + b2])

def piecewise_linear(x, x0, y0, k1, k2):
    x0 = 0.4
    y0 = 1100
    return np.piecewise(x, [x < x0], [lambda x: k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])


# def piecewise_linear(x, k1, k2, x0, y0): # need to be improve!!!!!!!!!!!!!!!
#     return np.piecewise(x, [x < 1100], [lambda x:k1*x + 0.5-k1*1100, lambda x:k2*x + 0.5-k2*1100])

def pol2(x, a, b, c):
    """
    Polynomial function
    """
    return a*x + b*x**2+c

def pol3(x, a, b, c,d):
    """
    Polynomial function
    """
    return a*x + b*x**2 +c*x**3 +d

def pol4(x, a, b, c,d,h):
    """
    Polynomial function
    """
    return a*x + b*x**2 +c*x**3+d*x**4+h

def lin(x, a, b):
    """
    linear function function
    """
    return a*x +b

def exp(x,a,b,c):    
    return a * np.exp(-b * x) +c


def mean_error(y_true, y_pred):
    return (y_pred - y_true).mean()

def lin2(x,a,b,c):
    x=x.T
    return a + b*x[0] + c*x[1]
    
    
def multi_pol3_lin(x, a, b, c, d, e):
    x = x.T
    return a*x[0] + b*x[0]**2 +c*x[0]**3 + d*x[1] +e

def multi_pol2_lin(x, a, b, c, d):
    x = x.T
    return a*x[0] + b*x[0]**2 +c*x[1] +d


 
def multi_pol3_lin_lin(x, a, b, c, d, e,f):
    x = x.T
    return a*x[0] + b*x[0]**2 +c*x[0]**3 + d*x[1] + e*x[2] +f
