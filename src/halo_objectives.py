import numpy as np
from autograd import numpy as agnp
from autograd import grad 

def softmax(x):
    '''From https://gist.github.com/stober/1946926'''
    e_x = agnp.exp(x - agnp.max(x))
    out = e_x / e_x.sum()
    return out

def tv(weights,lag,pixelvector):
    flux = agnp.dot(softmax(weights).T,pixelvector)
    diff = agnp.sum(agnp.abs(flux[lag:] - flux[:(-lag)]))
    return diff/agnp.mean(flux)

def tv_o2(weights,lag,pixelvector):
    flux = agnp.dot(softmax(weights).T,pixelvector)
    diff = agnp.sum(agnp.abs(2.*flux[lag:(-lag)] - flux[(2*lag):] - flux[:(-2*lag)]))
    return diff/agnp.mean(flux)

def l2v(weights,lag,pixelvector):
    flux = agnp.dot(softmax(weights).T,pixelvector)
    diff = agnp.sum((flux[lag:] - flux[:(-lag)])**2)
    return diff/agnp.mean(flux)

def l3v(weights,lag,pixelvector):
    flux = agnp.dot(softmax(weights).T,pixelvector)
    diff = agnp.sum(agnp.abs(flux[lag:] - flux[:(-lag)])**3)
    return diff/agnp.mean(flux)

def owl(weights,pixelvector):
    # this is strictly worse than all other options
    flux = agnp.dot(softmax(weights).T,pixelvector)
    diff = agnp.std(flux)
    return diff/agnp.mean(flux)

available = [tv,tv_o2,l2v,l3v]
mapping = {avail.__name__: avail for avail in available}
