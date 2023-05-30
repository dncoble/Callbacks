import numpy as np
"""
Metrics to evaluate model prediction. Calculation should be kind with
respect to array shape, generally meant to work for flattened arrays.

Tested with numpy 1.19.5
"""
""" signal to noise ratio """
def snr(sig, pred, dB=True):
    noise = sig - pred
    a_sig = np.sqrt(np.mean(np.square(sig)))
    a_noise = np.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*np.log10(snr)
""" root mean squared error """
def rmse(sig, pred, squared=False):
    error = sig - pred
    num = np.sum(np.square(error))
    denom = np.size(sig)
    e = num/denom
    if(squared):
        return e
    return np.sqrt(e)
""" root relative squared error """
def rrse(sig, pred):
    error = sig - pred
    mean = np.mean(sig)
    num = np.sum(np.square(error))
    denom = np.sum(np.square(sig-mean))
    return np.sqrt(num/denom)
""" normalized root mean squared error """
def nrmse(sig, pred):
    return rmse(sig, pred)/(np.max(sig)-np.min(sig))
""" time response assurance criterion """
def trac(sig, pred):
    num = np.square(np.dot(sig, pred))
    denom = np.dot(sig, sig) * np.dot(pred, pred)
    return num/denom
""" mean absolute error """
def mae(sig, pred):
    return np.sum(np.abs(sig-pred))/sig.size