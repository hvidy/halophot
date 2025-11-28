import jax
from jax import numpy as np

def softmax(x):
    '''From https://gist.github.com/stober/1946926'''
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def make_flux(weights,pixelvector):
    flux = np.dot(softmax(weights).T,pixelvector)
    return flux/np.mean(flux)

def tv(weights,lag,pixelvector):
    flux = make_flux(weights,pixelvector)
    diff = np.sum(np.abs(flux[lag:] - flux[:(-lag)]))
    return diff

def tv_exp(weights,lag,pixelvector):
    flux = make_flux(weights,pixelvector)

    flux_mat = np.abs(flux[:,np.newaxis] - flux[np.newaxis,:])
    lag_sum = np.sum(flux_mat, axis=0)
    return np.sum(np.exp(-np.arange(len(lag_sum))/lag)/lag * lag_sum)

def tv_o2(weights,lag,pixelvector):
    flux = make_flux(weights,pixelvector)
    diff = np.sum(np.abs(2.*flux[lag:(-lag)] - flux[(2*lag):] - flux[:(-2*lag)]))
    return diff/np.mean(flux)

def l2v(weights,lag,pixelvector):
    flux = make_flux(weights,pixelvector)
    diff = np.sum((flux[lag:] - flux[:(-lag)])**2)
    return diff

def l3v(weights,lag,pixelvector):
    flux = make_flux(weights,pixelvector)
    diff = np.sum(np.abs(flux[lag:] - flux[:(-lag)])**3)
    return diff

def owl(weights,pixelvector):
    # for halophot this is strictly worse than all other options
    flux = make_flux(weights,pixelvector)
    diff = np.std(flux)
    return diff/np.mean(flux)

available = [tv,tv_o2,l2v,l3v,owl,tv_exp]
mapping = {avail.__name__: avail for avail in available}