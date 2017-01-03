import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.table import Table
import scipy.optimize as optimize
import fitsio
from time import time as clock
import astropy.table
import theano
import theano.tensor as T
from theano import pp
from theano import In
from k2sc.utils import sigma_clip

'''-----------------------------------------------------------------
halo_tools.py 

In this package we include all the functions that are necessary for
halo photometry in Python.

-----------------------------------------------------------------'''

# =========================================================================
# =========================================================================

def print_time(t):
		if t>3600:
			print 'Time taken: %d h %d m %3f s'\
			% (np.int(np.floor(t/3600)), np.int(np.floor(np.mod(t,3600)/60)),np.mod(t,60))
		elif t>60:
			print 'Time taken: %d m %3f s' % (np.int(np.floor(np.mod(t,3600)/60)),np.mod(t,60) )
		else:
			print 'Time taken: %3f s' % t

# =========================================================================
# =========================================================================

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

def censor_tpf(tpf,ts,thresh=0.8,minflux=100.,do_quality=False):
	'''Throw away bad pixels and bad cadences'''

	dummy = tpf.copy()
	tsd = ts.copy()
	maxflux = np.nanmax(tpf)

	# find bad pixels

	if do_quality:

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
	tsd = tsd[np.all(np.isfinite(pixels),axis=0)]
	pixels = pixels[:,np.all(np.isfinite(pixels),axis=0)]
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

## it seems to be faster than using np.diff?

def diff_1(z):
	return np.sum(np.abs(z[1:-1]-np.roll(z[1:-1],1)))

def diff_2(z):
	return np.sum(np.abs(2*z[1:-1]-np.roll(z[1:-1],1)-np.roll(z[1:-1],-1)))


def tv_tpf(pixelvector,order=1,w_init=None,maxiter=101,analytic=False):
	'''Use first order for total variation on gradient, second order
	for total second derivative'''

	npix = np.shape(pixelvector)[0]
	cons = ({'type': 'eq', 'fun': lambda z: z.sum() - 1.})
	bounds = npix*((0,1),)

	if w_init is None:
		w_init = np.ones(npix)/np.float(npix)

	if analytic:
		w = T.dvector('w')
		p = T.dmatrix('p')
		ff = T.dot(T.nnet.softmax(w),p)
		ffd = T.roll(ff,1)

		if order == 1:
			diff = T.sum(T.abs_(ff-ffd))/T.mean(ff)
		elif order == 2:
			ffd2 = T.roll(ff,-1)
			diff = T.sum(T.abs_(2*ff-ffd-ffd2))/T.mean(ff)

		gw = T.grad(diff, w)
		# hw = T.hessian(diff,w)

		dtv = theano.function([w,In(p,value=pixelvector)],gw)
		tvf = theano.function([w,In(p,value=pixelvector)],diff)
		# hesstv = theano.function([w,In(p,value=pixelvector)],hw)

		res = optimize.minimize(tvf, w_init, method='L-BFGS-B', jac=dtv, 
			options={'disp': False,'maxiter':maxiter})
		w_best = np.exp(res['x'])/np.sum(np.exp(res['x'])) # softmax

		lc_first_try = np.dot(w_best.T,pixelvector)

		print 'Sigma clipping'

		good = sigma_clip(lc_first_try)

		print 'Clipping %d bad points' % np.sum(~good)

		pixels_masked, ts_masked = pixelvector[good,:], ts[good]

		dtv = theano.function([w,In(p,value=pixels_masked)],gw)
		tvf = theano.function([w,In(p,value=pixels_masked)],diff)

		res = optimize.minimize(tvf, w_init, method='L-BFGS-B', jac=dtv, 
			options={'disp': False,'maxiter':maxiter})
		w_best = np.exp(res['x'])/np.sum(np.exp(res['x'])) # softmax

	else:
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
	thresh=0.8,minflux=100.,consensus=False,analytic=False):
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
	if consensus:			
		assert sub>1, "Must be subsampled to use consensus"
		print 'Subsampling by a factor of', sub

		weights = np.zeros(pixels.shape[0])
		opt_lcs = np.zeros((pixels[::sub,:].shape[1],sub))

		if random_init:
			w_init = np.random.rand(pixels[::sub,:].shape[0])
			w_init /= np.sum(w_init)

		for j in range(sub):
			pixels_sub = pixels[j::sub,:]
			### now calculate the halo 
			print 'Calculating weights'

			weights[j::sub], opt_lcs[:,j] = tv_tpf(pixels_sub,order=order,
				maxiter=maxiter,w_init=w_init,analytic=analytic)
			print 'Calculated weights!'

		norm_lcs = opt_lcs/np.nanmedian(opt_lcs,axis=0)
		opt_lc = np.nanmean(norm_lcs,axis=1)

	else:
		pixels_sub = pixels[::sub,:]
		print 'Subsampling by a factor of', sub

		### now calculate the halo 

		print 'Calculating weights'
		if random_init:
			w_init = np.random.rand(pixels_sub.shape[0])
			w_init /= np.sum(w_init)

		weights, opt_lc = tv_tpf(pixels_sub,order=order,maxiter=maxiter,
			w_init=w_init,analytic=analytic)
		print 'Calculated weights!'

	# opt_lc = np.dot(weights.T,pixels_sub)
	ts['corr_flux'] = opt_lc

	if sub == 1:
		pixelmap.ravel()[mapping] = weights
		return tpf, ts, weights, pixelmap
	elif consensus:
		pixelmap.ravel()[mapping] = weights/float(sub)
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