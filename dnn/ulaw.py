
import numpy as np
import math

def ulaw2lin(u):
    s = np.sign(u)
    u = np.abs(u)
    return s*(np.exp(u/128.*math.log(256))-1)/255


def lin2ulaw(x):
    s = np.sign(x)
    x = np.abs(x)
    u = (s*(128*np.log(1+255*x)/math.log(256)))
    u = np.round(u)
    return u.astype('int16')
