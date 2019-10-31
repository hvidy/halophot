import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import scipy.optimize as optimize
import fitsio
from time import time as clock
from SuzPyUtils.norm import medsig
from psf_sim import *
from k2sc.cdpp import cdpp

import matplotlib as mpl

mpl.style.use('seaborn-colorblind')

#To make sure we have always the same matplotlib settings
#(the ones in comments are the ipython notebook settings)

mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
mpl.rcParams['font.size']=18               #10 
mpl.rcParams['savefig.dpi']= 200             #72 
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12


fname = '../EPIC_211309989_mast.fits' # point this path to your favourite K2SC light curve
lc = Table.read(fname)

x, y = lc['x'][150:1550], lc['y'][150:1550] # copy in the xy variations from real data 

ncad = np.size(x)

'''------------------------
Generate a toy light curve
------------------------'''

t = np.linspace(0,100,ncad)

f = 20*np.ones(ncad) + np.sin(t/6.) # make this whatever function you like! 
f[400:500] *= 0.990 # toy transit


'''------------------------
Define a PSF and aperture
------------------------'''

sizes = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])


sigs_raw = 1.*np.zeros_like(sizes)
sigs_1 = 1.*np.zeros_like(sizes)
sigs_2 = 1.*np.zeros_like(sizes)
times_1 = 1.*np.zeros_like(sizes)
times_2 = 1.*np.zeros_like(sizes)

widths = [1.,2.5,5.]

start = clock()

for width in widths:
	print 'Doing witdth',width
	for jj, nn in enumerate(sizes):
		print 'Trying window size',nn
		nx, ny = nn, nn
		npix = nx*ny
		pixels = np.zeros((nx,ny))

		psf = gaussian_psf(pixels,x[0],y[0],width)

		'''------------------------
		Simulate data
		------------------------'''

		tpf = np.zeros((nx,ny,ncad))
		sensitivity = 1-0.1*np.random.rand(nx,ny)
		white = 0

		for j in range(ncad):
		    tpf[:,:,j] = f[j]*gaussian_psf(pixels,x[j],y[j],width)*sensitivity + np.random.randn(nx,ny)*white

		pixelvectors = np.reshape(tpf,(nx*ny,ncad))

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

		tic = clock()
		    
		res1 = optimize.minimize(obj_1, w_init, method='SLSQP', constraints=cons, bounds = bounds,
		                        options={'disp': False})
		xbest_1 = res1['x']

		toc = clock()

		res2 = optimize.minimize(obj_2, xbest_1, method='SLSQP', constraints=cons, bounds = bounds,
		                        options={'disp': False})
		xbest_2 = res2['x']
		toc2 = clock()

		print 'Time taken for TV1:',(toc-tic)
		print 'Time taken for TV2:', (toc2-toc)

		lc_opt_1 = np.dot(xbest_1.T,pixelvectors)
		lc_opt_2 = np.dot(xbest_2.T,pixelvectors)

		raw_lc = np.sum(pixelvectors,axis=0)

		raw_lc /= np.nanmedian(raw_lc)
		lc_opt_1 /= np.nanmedian(lc_opt_1)
		lc_opt_2 /= np.nanmedian(lc_opt_2)

		ssr = cdpp(t,raw_lc-f/np.nanmedian(f)+1)
		ss1 = cdpp(t,lc_opt_1-f/np.nanmedian(f)+1)
		ss2 = cdpp(t,lc_opt_2-f/np.nanmedian(f)+1)

		sigs_raw[jj] = ssr
		sigs_1[jj] = ss1
		sigs_2[jj] = ss2 
		times_1[jj] = toc-tic
		times_2[jj] = toc2-toc

	finish = clock()
	print 'Done'
	print 'Time elapsed:',finish-start

	ticks = np.unique(sizes[sizes%5==0])

	plt.figure(0)
	plt.clf()

	plt.plot(sizes,sigs_raw,'.-',label='Raw')
	plt.plot(sizes,sigs_1,'.-',label='TV1')
	plt.plot(sizes,sigs_2,'.-',label='TV2')
	plt.ylabel('CDPP (ppm)')
	plt.xlabel('Window Size (pix)')
	plt.yscale('log')
	plt.title(r'Precision of TV: Gaussian PSF, $\sigma$=%.1f' % width)
	plt.xticks(ticks)
	plt.legend()
	plt.savefig('cdpp_w%.1f_s%d.png' % (width,np.max(sizes)))

	plt.figure(1)
	plt.clf()

	plt.plot(sizes,times_1,'.-',label='TV1')
	plt.plot(sizes,times_2,'.-',label='TV2')
	plt.axhline(8)
	plt.ylabel('Seconds')
	plt.xlabel('Window Size (pix)')
	plt.yscale('log')
	plt.title('Time Taken')
	plt.xticks(ticks)
	plt.legend()
	plt.savefig('times_w%.1f_s%d.png' % (width,np.max(sizes)))
