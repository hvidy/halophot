import numpy as np
import matplotlib.pyplot as plt

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
