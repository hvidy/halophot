import numpy as np
from autograd import numpy as agnp
from autograd import grad 
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.table import Table
import scipy.optimize as optimize
from astropy.io import fits
from time import time as clock
import astropy.table

import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

'''-----------------------------------------------------------------
halo_tools.py 

In this package we include all the functions that are necessary for
halo photometry in Python.

-----------------------------------------------------------------'''

def softmax(x):
	'''From https://gist.github.com/stober/1946926'''
	e_x = agnp.exp(x - agnp.max(x))
	out = e_x / e_x.sum()
	return out

# =========================================================================
# =========================================================================


def sigma_clip(a, max_iter=10, max_sigma=5, separate_masks=False, mexc=None):
    """Iterative sigma-clipping routine that separates not finite points, and down- and upwards outliers.

    from k2sc, authors: Aigrain, Parviainen & Pope
    """
    mexc  = isfinite(a) if mexc is None else isfinite(a) & mexc
    mhigh = ones_like(mexc)
    mlow  = ones_like(mexc)
    mask  = ones_like(mexc)

    i, nm = 0, None
    while (nm != mask.sum()) and (i < max_iter):
        mask = mexc & mhigh & mlow
        nm = mask.sum()
        med, sig = medsig(a[mask])
        mhigh[mexc] = a[mexc] - med <  max_sigma*sig
        mlow[mexc]  = a[mexc] - med > -max_sigma*sig
        i += 1

    if separate_masks:
        return mlow, mhigh
    else:
        return mlow & mhigh

# =========================================================================
# =========================================================================


def print_time(t):
		if t>3600:
			print('Time taken: %d h %d m %3f s'\
			% (np.int(np.floor(t/3600)), np.int(np.floor(np.mod(t,3600)/60)),np.mod(t,60)))
		elif t>60:
			print( 'Time taken: %d m %3f s' % (np.int(np.floor(np.mod(t,3600)/60)),np.mod(t,60) ))
		else:
			print( 'Time taken: %3f s' % t)

# =========================================================================
# =========================================================================

def read_tpf(fname):
	target_fits = fits.open(fname)

	tpf = target_fits[1].data['FLUX'][:]

	t, x, y = target_fits[1].data['TIME'][:], target_fits[1].data['POS_CORR1'][:], target_fits[1].data['POS_CORR2'][:]
	cad = target_fits[1].data['CADENCENO'][:]
	quality = target_fits[1].data['QUALITY'][:].astype('int32')

	ts = Table({'time':t,
				'cadence':cad,
				'x':x,
				'y':y,
				'quality':quality})


	return tpf, ts

# =========================================================================
# =========================================================================

def censor_tpf(tpf,ts,thresh=0.8,minflux=100.,do_quality=True):
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
	
	xc, yc = np.nanmedian(ts['x']), np.nanmedian(ts['y'])

	if np.sum(np.isfinite(ts['x']))>=0.8*tsd['x'].shape[0]:
		rr = np.sqrt((tsd['x']-xc)**2 + (tsd['y']-yc)**2)
		goodpos = (rr<5) * np.isfinite(tsd['x']) * np.isfinite(tsd['y'])
		dummy = dummy[goodpos,:,:] # some campaigns have a few extremely bad cadences

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

# =========================================================================
# =========================================================================

def get_slice(tpf,ts,start,stop):
	return tpf[start:stop,:,:], ts[start:stop]

# =========================================================================
# =========================================================================

def get_annulus(tpf,rmin,rmax):
	xs, ys = np.arange(tpf.shape[2])-tpf.shape[2]/2.,np.arange(tpf.shape[1])-tpf.shape[1]/2.
	xx, yy = np.meshgrid(xs,ys)
	rr = np.sqrt(xx**2 + yy **2)
	mask = (rr>rmax) + (rr<rmin)
	tpf[:,mask] = np.nan
	return tpf

# =========================================================================
# =========================================================================

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

# =========================================================================
# =========================================================================

def tv_tpf(pixelvector,order=1,w_init=None,maxiter=101,analytic=False,sigclip=False):
	'''
	This is the main function here - once you have loaded the data, pass it to this
	to do a TV-min light curve.

	Keywords

	order: int
		Run nth order TV - ie first order is L1 norm on first derivative,
		second order is L1 norm on second derivative, etc.
		This is part of the Pock generalized TV scheme, so that
		1st order gives you piecewise constant functions,
		2nd order gives you piecewise affine functions, etc. 
		Currently implemented only up to 2nd order in numerical, 1st in analytic!
		We recommend first order very strongly.
	maxiter: int
		Number of iterations to optimize. 101 is default & usually sufficient.
	w_init: None or array-like.
		Initialize weights with a particular weight vector - useful if you have
		already run TV-min and want to update, but otherwise set to None 
		and it will have default initialization.
	random_init: Boolean
		If False, and w_init is None, it will initialize with uniform weights; if True, it
		will initialize with random weights. False is usually better.
	thresh: float
		A float greater than 0. Pixels less than this fraction of the maximum
		flux at any pixel will be masked out - this is to deal with saturation.
		Because halo is usually intended for saturated stars, the default is 0.8, 
		to deal with saturated pixels. If your star is not saturated, set this 
		greater than 1.0. 
	consensus: Boolean
		If True, this will subsample the pixel space, separately calculate halo time 
		series for eah set of pixels, and merge these at the end. This is to check
		for validation, but is typically not useful, and is by default set False.
	analytic: Boolean
		If True, it will optimize the TV with autograd analytic derivatives, which is
		several orders of magnitude faster than with numerical derivatives. This is 
		by default True but you can run it numerically with False if you prefer.
	sigclip: Boolean
		If True, it will iteratively run the TV-min algorithm clipping outliers.
		Use this for data with a lot of outliers, but by default it is set False.
	'''

	npix = np.shape(pixelvector)[0]
	cons = ({'type': 'eq', 'fun': lambda z: z.sum() - 1.})
	bounds = npix*((0,1),)

	if w_init is None:
		w_init = np.ones(npix)/np.float(npix)

	if analytic: 
		print('Using Analytic Derivatives')
		# only use first order, it appears to be strictly better

		def tv_soft(weights):
			flux = agnp.dot(softmax(weights).T,pixelvector)
			diff = agnp.sum(agnp.abs(flux[1:] - flux[:-1]))
			return diff/agnp.mean(flux)

		gradient = grad(tv_soft)

		res = optimize.minimize(tv_soft, w_init, method='L-BFGS-B', jac=gradient, 
			options={'disp': False,'maxiter':maxiter})

		w_best = softmax(res['x']) # softmax

		lc_first_try = np.dot(w_best.T,pixelvector)

		if sigclip:
			print('Sigma clipping')

			good = sigma_clip(lc_first_try,max_sigma=3.5)


			if np.sum(~good) > 0:
				print('Clipping %d bad points' % np.sum(~good))

				pixels_masked = pixelvector[:,good]

				def tv_masked(weights):
					flux = agnp.dot(softmax(weights).T,pixels_masked)
					diff = agnp.sum(agnp.abs(flux[1:] - flux[:-1]))
					return diff/agnp.mean(flux)

				gradient_masked = grad(tv_masked)

				res = optimize.minimize(tv_masked, w_init, method='L-BFGS-B', jac=gradient_masked, 
					options={'disp': False,'maxiter':maxiter})

				w_best = softmax(res['x']) # softmax
			else:
				print('No outliers found, continuing')
		else:
			pass

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
			print('Failed to converge well! Rescaling.')
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

# =========================================================================
# =========================================================================

def do_lc(tpf,ts,splits,sub,order,maxiter=101,w_init=None,random_init=False,
	thresh=0.8,minflux=100.,consensus=False,analytic=False,sigclip=False):
	### get a slice corresponding to the splits you want
	if splits[0] is None and splits[1] is not None:
		print('Taking cadences from beginning to',splits[1])
	elif splits[0] is not None and splits[1] is None:
		print('Taking cadences from', splits[0],'to end')
	elif splits[0] is None and splits[1] is None:
		print('Taking cadences from beginning to end')
	else:
		print('Taking cadences from', splits[0],'to',splits[1])

	tpf, ts = get_slice(tpf,ts,splits[0],splits[1])

	### now throw away saturated columns, nan pixels and nan cadences

	pixels, ts, mapping = censor_tpf(tpf,ts,thresh=thresh,minflux=minflux)
	pixelmap = np.zeros((tpf.shape[2],tpf.shape[1]))
	print('Censored TPF')

	### subsample
	if consensus:			
		assert sub>1, "Must be subsampled to use consensus"
		print('Subsampling by a factor of %d' % sub)

		weights = np.zeros(pixels.shape[0])
		opt_lcs = np.zeros((pixels[::sub,:].shape[1],sub))

		if random_init:
			w_init = np.random.rand(pixels[::sub,:].shape[0])
			w_init /= np.sum(w_init)

		for j in range(sub):
			pixels_sub = pixels[j::sub,:]
			### now calculate the halo 
			print('Calculating weights')

			weights[j::sub], opt_lcs[:,j] = tv_tpf(pixels_sub,order=order,
				maxiter=maxiter,w_init=w_init,analytic=analytic,sigclip=sigclip)
			print('Calculated weights!')

		norm_lcs = opt_lcs/np.nanmedian(opt_lcs,axis=0)
		opt_lc = np.nanmean(norm_lcs,axis=1)

	else:
		pixels_sub = pixels[::sub,:]
		print('Subsampling by a factor of %d' % sub)

		### now calculate the halo 

		print('Calculating weights')
		if random_init:
			w_init = np.random.rand(pixels_sub.shape[0])
			w_init /= np.sum(w_init)

		weights, opt_lc = tv_tpf(pixels_sub,order=order,maxiter=maxiter,
			w_init=w_init,analytic=analytic)
		print('Calculated weights!')

	# opt_lc = np.dot(weights.T,pixels_sub)
	ts['corr_flux'] = opt_lc

	if sub == 1:
		pixelmap.ravel()[mapping] = weights
		return tpf, ts, weights, pixelmap, pixels_sub
	elif consensus:
		pixelmap.ravel()[mapping] = weights/float(sub)
		return tpf, ts, weights, pixelmap, pixels_sub
	else:
		pixelmap.ravel()[mapping[0][::sub]] = weights
		return tpf, ts, weights, pixelmap, pixels_sub

'''-----------------------------------------------------------------
The cuts for Campaign 4 are

0:550
550:2200
2200:
-----------------------------------------------------------------'''