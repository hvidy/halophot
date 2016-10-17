import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
	cad = target_fits[1]['CADENCENO'][:]
	quality = target_fits[1]['QUALITY'][:].astype('int32')

	ts = Table({'time':t,
				'cadence':cad,
				'x':x,
				'y':y,
				'quality':quality})

	return tpf, ts

def censor_tpf(tpf,ts,thresh=0.8,minflux=100.):
	'''Throw away bad pixels and bad cadences'''

	dummy = tpf.copy()
	tsd = ts.copy()
	maxflux = np.nanmax(tpf)

	# find bad pixels

	m = (ts['quality'] == 0) # get bad quality 
	dummy = dummy[m,:,:]
	tsd = tsd[m]

	dummy[dummy<0] = 0 # just as a check!

	saturated = np.nanmax(dummy,axis=0) > (thresh*maxflux)
	dummy[:,saturated] = np.nan 

	no_flux = np.nanmin(dummy,axis=0) < minflux
	dummy[:,no_flux] = np.nan

	# then pick only pixels which are mostly good

	pixels = np.reshape(dummy.T,((tpf.shape[1]*tpf.shape[2]),dummy.shape[0]))
	indic = np.array([np.sum(np.isfinite(pixels[j,:])) 
		for j in range(pixels.shape[0])])
	pixels = pixels[indic>60,:]

	# indic_cad = np.array([np.sum(np.isfinite(pixels[:,j])) 
	# 	for j in range(pixels.shape[1])])

	# pixels = pixels[:,indic_cad>200]
	# ts = ts[indic_cad>200]

	# this should get all the nans but if not just set them to 0
	pixels[~np.isfinite(pixels)] = 0

	return pixels,tsd, np.where(indic>60)

def get_slice(tpf,ts,start,stop):
	return tpf[start:stop,:,:], ts[start:stop]

def get_annulus(tpf,rmin,rmax):
	xs, ys = np.arange(tpf.shape[2])-tpf.shape[2]/2.,np.arange(tpf.shape[1])-tpf.shape[1]/2.
	xx, yy = np.meshgrid(xs,ys)
	rr = np.sqrt(xx**2 + yy **2)
	mask = (rr>rmax) + (rr<rmin)
	tpf[:,mask] = np.nan
	return tpf

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

def grad_1(w,pixels):
	flux = np.dot(weights.T,pixelvector)
	flux /= np.nanmedian(flux)
	diffs = np.abs(z[1:-1]-np.roll(z[1:-1],1))
	return np.dot(pixels,(diffs-np.roll(diffs,1)))

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

	if 'Positive directional derivative for linesearch' in res['message']:
		print 'Failed to converge well! Rescaling.'
		if order==1:
			def obj(weights):
				flux = np.dot(weights.T,pixelvector)
				flux /= np.nanmedian(flux)
				return diff_1(flux)/10.

		elif order==2:
			def obj(weights):
				flux = np.dot(weights.T,pixelvector)
				flux/= np.nanmedian(flux)
				return diff_2(flux)/10.
		w_init = np.random.rand(npix)
		w_init /= w_init.sum()
		res = optimize.minimize(obj, w_init, method='SLSQP', constraints=cons, 
			bounds = bounds, options={'disp': True,'maxiter':maxiter})
	
	w_best = res['x']
	lc_opt = np.dot(w_best.T,pixelvector)
	return w_best, lc_opt

def do_lc(tpf,ts,splits,sub,order,maxiter=101,w_init=None,random_init=False,
	thresh=0.8,minflux=100.):
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

	pixels, ts, mapping = censor_tpf(tpf,ts,thresh=thresh,minflux=minflux)
	pixelmap = np.zeros((tpf.shape[2],tpf.shape[1]))
	print 'Censored TPF'

	### subsample

	pixels_sub, ts_sub = pixels[::sub,:], ts[::sub]
	print 'Subsampling by a factor of', sub

	### now calculate the halo 

	print 'Calculating weights'
	if random_init:
		w_init = np.random.rand(pixels_sub.shape[0])
		w_init /= np.sum(w_init)

	weights, opt_lc = tv_tpf(pixels_sub,order=order,maxiter=maxiter,w_init=w_init)
	print 'Calculated weights!'

	ts['corr_flux'] = opt_lc

	if sub == 1:
		pixelmap.ravel()[mapping] = weights
		return tpf, ts, weights, pixelmap

	else:
		pixelmap.ravel()[mapping[0][::sub]] = weights
		return tpf, ts, weights, pixelmap

'''-----------------------------------------------------------------
The cuts for Campaign 4 are

0:550
550:2200
2200:
-----------------------------------------------------------------'''