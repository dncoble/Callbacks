import numpy as np
"""
Features used for data analysis. Feature calculation should be kind with
respect to array shape, generally meant to work for flattened arrays.

Tested with numpy 1.19.5
"""
def maximum(signal, absolute=True):
    if(absolute):
        return max(np.max(signal), -np.max(-signal))
    else:
        return np.max(signal)
""" mean of absolute value """
def absolutemean(signal):
    return np.mean(np.abs(signal))
""" root mean squared """
def rms(signal):
    return np.sqrt(np.sum(np.square(signal))/signal.size)
### data statistics related
""" standard deviation """
def std(signal, squared=False):
    m = np.mean(signal)
    s_ = np.sum(np.square(signal - m))
    s_ /= signal.size
    if(squared):
        return s_
    return np.sqrt(s_)
""" skewness """
def skewness(signal):
    N = signal.size
    m = np.mean(signal)
    E = np.sum((signal-m)**3)
    return E/((N-1)*std(signal)**3)
""" kurtosis """
def kurtosis(signal):
    N = signal.size
    m = np.mean(signal)
    E = np.sum((signal-m)**4)
    return E/((N-1)*std(signal)**4)
### sinusoidal wave shape related
""" crest factor """
def crestfactor(signal):
    return maximum(signal)/rms(signal)
""" shape factor """
def shapefactor(signal):
    return rms(signal)/absolutemean(signal)
""" impulse factor """
def impulsefactor(signal):
    return maximum(signal)/absolutemean(signal)