import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import scipy.optimize as optimize
import fitsio
from time import time as clock
import astropy.table

'''-----------------------------------------------------------------
halo_tools.py 

In this package we include all the functions that are necessary for
halo photometry in Python.

-----------------------------------------------------------------'''

def read_tpf(fname):
	target_fits = fitsio.FITS(fname)

	tpf = target_fits[1]['FLUX'][:]

	t, x, y = target_fits[1]['TIME'][:], target_fits[1]['POS_CORR1'][:], target_fits[1]['POS_CORR2'][:]
	quality = target_fits[1]['QUALITY'][:].astype('int32')

	ts = Table({'time':t,
				'x':x,
				'y':y,
				'quality':quality})

	return tpf, ts

def censor_tpf(tpf,ts,thresh=0.8):
	'''Throw away bad pixels and bad cadences'''

	# first find bad pixels
	maxflux = np.nanmax(tpf)

	for j in range(tpf.shape[1]):
		for k in range(tpf.shape[2]):
			if np.nanmax(tpf[:,j,k]) > (thresh*maxflux):
				tpf[:,j,k] = np.nan
			elif np.nanmin(tpf[:,j,k]) < 100.:
				tpf[:,j,k] = np.nan

	# find bad pixels

	pixels = np.reshape(tpf.T,((tpf.shape[1]*tpf.shape[2]),tpf.shape[0]))
	indic = np.array([np.sum(np.isfinite(pixels[j,:])) 
		for j in range(pixels.shape[0])])
	pixels = pixels[indic>60,:]

	# next find bad cadences

	m = (ts['quality'] == 0) # get bad quality 
	pixels = pixels[:,m]
	ts = ts[m]

	# indic_cad = np.array([np.sum(np.isfinite(pixels[:,j])) 
	# 	for j in range(pixels.shape[1])])

	# pixels = pixels[:,indic_cad>200]
	# ts = ts[indic_cad>200]

	# this should get all the nans but if not just set them to 0
	pixels[~np.isfinite(pixels)] = 0

	return pixels,ts, np.where(indic>60)

def get_slice(tpf,ts,start,stop):
	return tpf[start:stop,:,:], ts[start:stop]

def stitch(tslist):
	# key idea is to match GP values at the edge
	final = np.nanmedian(tslist[0]['corr_flux'][-5:])
	newts = tslist[0].copy()
	for tsj in tslist[1:]:
		initial = np.nanmedian(tsj['corr_flux'][:5])
		tsj['corr_flux'] += final-initial
		final = np.nanmedian(tsj['corr_flux'][-5:])
		newts = astropy.table.vstack([newts,tsj])
	return newts 

'''-----------------------------------------------------------------
In this section we include the actual detrending code.
-----------------------------------------------------------------'''

def diff_1(z):
	return np.sum(np.abs(z[1:-1]-np.roll(z[1:-1],1)))

def diff_2(z):
	return np.sum(np.abs(2*z[1:-1]-np.roll(z[1:-1],1)-np.roll(z[1:-1],2)))

def tv_tpf(pixelvector,order=1,w_init=None,maxiter=101):
	'''Use first order for total variation on gradient, second order
	for total second derivative'''

	npix = np.shape(pixelvector)[0]
	cons = ({'type': 'eq', 'fun': lambda z: z.sum() - 1.})
	bounds = npix*((0,1),)
	if w_init is None:
		w_init = np.ones(npix)/np.float(npix)

	if order==1:
		def obj(weights):
			flux = np.dot(weights.T,pixelvector)
			flux /= np.nanmedian(flux)
			return diff_1(flux)

	elif order==2:
		def obj(weights):
			flux = np.dot(weights.T,pixelvector)
			flux/= np.nanmedian(flux)
			return diff_2(flux)

	res = optimize.minimize(obj, w_init, method='SLSQP', constraints=cons, 
		bounds = bounds, options={'disp': True,'maxiter':maxiter})
	
	w_best = res['x']
	lc_opt = np.dot(w_best.T,pixelvector)
	return w_best, lc_opt

def do_lc(tpf,ts,splits,sub,order,maxiter=101):
	### get a slice corresponding to the splits you want
	if splits[0] is None and splits[1] is not None:
		print 'Taking cadences from beginning to',splits[1]
	elif splits[0] is not None and splits[1] is None:
		print 'Taking cadences from', splits[0],'to end'
	elif splits[0] is None and splits[1] is None:
		print 'Taking cadences from beginning to end'
	else:
		print 'Taking cadences from', splits[0],'to',splits[1]

	tpf, ts = get_slice(tpf,ts,splits[0],splits[1])

	### now throw away saturated columns, nan pixels and nan cadences

	pixels, ts, mapping = censor_tpf(tpf,ts,thresh=0.8)
	pixelmap = np.zeros((tpf.shape[1],tpf.shape[2]))
	print 'Censored TPF'

	### subsample

	pixels_sub, ts_sub = pixels[::sub,:], ts[::sub]
	print 'Subsampling by a factor of', sub

	### now calculate the halo 

	print 'Calculating weights'

	weights, opt_lc = tv_tpf(pixels_sub,order=order,maxiter=maxiter)
	print 'Calculated weights!'

	ts['corr_flux'] = opt_lc

	if sub == 1:
		pixelmap.ravel()[mapping] = weights
		return tpf, ts, weights, pixelmap

	else:
		return tpf, ts, weights, pixelmap

'''-----------------------------------------------------------------
The cuts for Campaign 4 are

0:550
550:2200
2200:
-----------------------------------------------------------------'''

