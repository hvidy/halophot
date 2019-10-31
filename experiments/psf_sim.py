import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

'''Functions for PSF simulation'''

def gaussian_psf(pixels,xp,yp,width):
    # make rr array
    npix = np.shape(pixels[0])[0]
    xx,yy = np.meshgrid(np.arange(npix)-npix/2.,np.arange(npix)-npix/2.)
    rr2 = (xx-xp)**2 + (yy-yp)**2
    
    return 1./(width*np.sqrt(2.*np.pi))*np.exp(-0.5*(rr2/width**2))

    '''------------------------------------------
Define functions here - all have equiv in
the main halophot package
------------------------------------------'''

def diff_1(z):
    return np.sum(np.abs(z-np.roll(z,1)))

def diff_2(z):
    return np.sum(np.abs(2*z-np.roll(z,1)-np.roll(z,2)))


def tv_tpf(pixelvector):
    npix = np.shape(pixelvector)[0]
    cons = ({'type': 'eq', 'fun': lambda z: z.sum() - 1.})
    bounds = npix*((0,1),)
    w_init = np.ones(npix)/np.float(npix)
    def objective_1(weights):
        flux = np.dot(weights.T,pixelvector)
        return diff_1(flux)/np.nanmedian(flux)
    res1 = optimize.minimize(objective_1, w_init, method='SLSQP', constraints=cons, bounds = bounds,
                        options={'disp': True})
    xbest_1 = res1['x']
    lc_opt_1 = np.dot(xbest_1.T,pixelvector)
    return xbest_1, lc_opt_1

def make_data(xs,ys,fs,params={'width':3,'nx':10,'ny':10,'white':0,'sensitivity':None}):
    nx = params['nx']
    ny = params['ny']

    npix = nx*ny
    width = params['width']
    white = params['white']

    ncad = np.size(xs)

    pixels = np.zeros((nx,ny))

    if params['sensitivity'] is None:
        sensitivity=1-0.1*np.random.rand(nx,ny)
    else:
        sensitivity = params['sensitivity']

    '''------------------------
    Simulate data
    ------------------------'''

    tpf = np.zeros((nx,ny,ncad))

    for j in range(ncad):
        tpf[:,:,j] = fs[j]*gaussian_psf(pixels,xs[j],ys[j],width)*sensitivity + np.random.randn(nx,ny)*white

    pixelvectors = np.reshape(tpf,(nx*ny,ncad))
    return pixelvectors


def do_sim(xs, ys, fs,params={'width':3,'nx':10,'ny':10,'white':0,'sensitivity':None}):


    pixelvectors = make_data(xs,ys,fs,params)

    '''------------------------
    Define objectives
    ------------------------'''


    def obj_1(weights):
        flux = np.dot(weights.T,pixelvectors)
        return diff_1(flux)/np.nanmedian(flux)

    def obj_2(weights):
    #     return np.dot(w.T,sigma_flux,w)
        flux = np.dot(weights.T,pixelvectors)
        return diff_2(flux)/np.nanmedian(flux)

    '''------------------------
    Reconstruct lightcurves
    ------------------------'''

    cons = ({'type': 'eq', 'fun': lambda z: z.sum() - 1.})
    bounds = (npix)*((0,1),)

    w_init = np.random.rand(npix)
    w_init /= np.sum(w_init)
    # w_init = np.ones(180)/180.
        
    res1 = optimize.minimize(obj_1, w_init, method='SLSQP', constraints=cons, bounds = bounds,
                            options={'disp': False})
    xbest_1 = res1['x']

    lc_opt_1 = np.dot(xbest_1.T,pixelvectors)

    raw_lc = np.sum(pixelvectors,axis=0)

    raw_lc /= np.nanmedian(raw_lc)
    lc_opt_1 /= np.nanmedian(lc_opt_1)

    return raw_lc, lc_opt_1 

def mad(a,b):
    '''Median absolute deviation'''
    return np.median(np.abs(a-b))
