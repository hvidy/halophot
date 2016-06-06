import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import scipy.optimize as optimize
import fitsio
from time import time as clock

'''-----------------------------------------------------------------
halo_tools.py 

In this package we include all the functions that are necessary for
halo photometry in Python.

-----------------------------------------------------------------'''

def read_tpf(fname):
	target_fits = fitsio.FITS(fname)

	tpf = target_fits[1]['FLUX'][:]

	t, x, y = target_fits[1]['TIME'][:], target_fits[1]['POS_CORR1'][:], target_fits[1]['POS_CORR2'][:]
	# quality = target_fits[1]['QUALITY'][:]
	# print quality

	ts = Table({'time':t,
				'x':x,
				'y':y})

	return tpf, ts

def censor_tpf(tpf,ts,thresh=0.8):
	'''Throw away bad pixels and bad cadences'''

	# first find bad pixels
	maxflux = np.nanmax(tpf)
	print tpf.shape
	print tpf

	for j in range(tpf.shape[1]):
		for k in range(tpf.shape[2]):
			if np.nanmax(tpf[:,j,k]) > (thresh*maxflux):
				tpf[:,j,k] = np.nan
			elif np.nanmin(tpf[:,j,k]) < 100.:
				tpf[:,j,k] = np.nan

	# next find bad cadences

	pixels = np.reshape(tpf.T,((tpf.shape[1]*tpf.shape[2]),tpf.shape[0]))
	indic = np.zeros(pixels.shape[0])
	for j in range(pixels.shape[0]):
		indic[j] = np.sum(np.isfinite(pixels[j,:]))

	pixels = pixels[indic>60,:]
	ts = ts[indic>60,:]

	indic_cad = np.zeros(pixels.shape[1])
	for j in range(newpixels.shape[1]):
		indic_cad[j] = np.sum(np.isfinite(pixels[:,j]))

	pixels = pixels[:,indic_cad>200]
	ts = ts[:,indic_cad>200]

	# this should get all the nans but if not just set them to 0
	pixels[~np.isfinite(pixels)] = 0

	return pixels,ts


def get_slice(tpf,ts,start,stop):
	return tpf[:,start:stop], ts[start:stop]

'''-----------------------------------------------------------------
In this section we include the actual detrending code.
-----------------------------------------------------------------'''

def diff_1(z):
	return np.sum(np.abs(z-np.roll(z,1)))

def diff_2(z):
	return np.sum(np.abs(2*z-np.roll(z,1)-np.roll(z,2)))

def tv_tpf(pixelvector,order=1):
	'''Use first order for total variation on gradient, second order
	for total second derivative'''

	npix = np.shape(pixelvector)[0]
	cons = ({'type': 'eq', 'fun': lambda z: z.sum() - 1.})
	bounds = npix*((0,1),)
	w_init = np.ones(npix)/np.float(npix)

	if order==1:
		def obj(weights):
			flux = np.dot(weights.T,pixelvector)
			return diff_1(flux)/np.nanmedian(flux)

	elif order==2:
		def obj(weights):
			flux = np.dot(weights.T,pixelvector)
			return diff_1(flux)/np.nanmedian(flux)

	res = optimize.minimize(obj, w_init, method='SLSQP', constraints=cons, bounds = bounds,
						options={'disp': True})
	xbest = res['x']
	lc_opt = np.dot(xbest.T,pixelvector)
	return xbest, lc_opt


'''-----------------------------------------------------------------
The cuts for Campaign 4 are

0:550
550:2200
2200:
-----------------------------------------------------------------'''

