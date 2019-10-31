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
colours = mpl.rcParams['axes.color_cycle']

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

# saturations = np.concatenate((range(5,25,1),range(25,400,10)))
saturations = np.arange(20)

sigs_raw = 1.*np.zeros(np.size(saturations))
sigs_1 = 1.*np.zeros(np.size(saturations))
sigs_2 = 1.*np.zeros(np.size(saturations))
times_1 = 1.*np.zeros(np.size(saturations))
times_2 = 1.*np.zeros(np.size(saturations))

width = 6.

nx, ny = 20, 20
npix = nx*ny
pixels = np.zeros((nx,ny))
psf = gaussian_psf(pixels,x[0],y[0],width)
tpf = np.zeros((nx,ny,ncad))
sensitivity = 1.
white = 0


'''------------------------
Simulate data
------------------------'''

for j in range(ncad):
    tpf[:,:,j] = f[j]*gaussian_psf(pixels,x[j],y[j],width)*sensitivity + np.random.randn(nx,ny)*white

pixelvectors_all = np.reshape(tpf,(npix,ncad))

all_inds = range(npix)

start = clock()

for jj,saturation in enumerate(saturations):
	print 'Saturating',saturation,'columns'
	# print 'Taking',sampling,'pixels'
	bad = all_inds[(nx/2-saturation/2):(nx/2+saturation/2)]
	tpf_masked = np.delete(tpf,bad,1)
	pixelvectors = np.reshape(tpf_masked,(tpf_masked.shape[0]*tpf_masked.shape[1],ncad))


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
	bounds = (pixelvectors.shape[0])*((0,1),)

	w_init = np.random.rand(pixelvectors.shape[0])
	w_init /= np.sum(w_init)
	# w_init = np.ones(180)/180.

	tic = clock()
	    
	res1 = optimize.minimize(obj_1, w_init, method='SLSQP', constraints=cons, bounds = bounds,
	                        options={'disp': False})
	xbest_1 = res1['x']

	toc = clock()

	# res2 = optimize.minimize(obj_2, xbest_1, method='SLSQP', constraints=cons, bounds = bounds,
	#                         options={'disp': False})
	# xbest_2 = res2['x']
	# toc2 = clock()

	print 'Time taken for TV1:',(toc-tic)
	# print 'Time taken for TV2:', (toc2-toc)

	lc_opt_1 = np.dot(xbest_1.T,pixelvectors)
	# lc_opt_2 = np.dot(xbest_2.T,pixelvectors)

	raw_lc = np.sum(pixelvectors,axis=0)

	raw_lc /= np.nanmedian(raw_lc)
	lc_opt_1 /= np.nanmedian(lc_opt_1)
	# lc_opt_2 /= np.nanmedian(lc_opt_2)

	ssr = cdpp(t,raw_lc-f/np.nanmedian(f)+1)
	ss1 = cdpp(t,lc_opt_1-f/np.nanmedian(f)+1)

	sigs_raw[jj] = ssr
	sigs_1[jj] = ss1
	times_1[jj] = toc-tic

finish = clock()
print 'Done'
print 'Time elapsed:',finish-start

# ticks = np.unique(sizes[sizes%5==0])

plt.figure(0)
plt.clf()

plt.plot(saturations,sigs_raw,'.-',label='Raw')
plt.plot(saturations,sigs_1,'.-',label='TV1')
plt.ylabel('CDPP (ppm)')
plt.xlabel('Saturated Columns')
plt.yscale('log')
plt.title(r'Precision of TV: Gaussian PSF, $\sigma$=%.1f' % width)
# plt.xticks(ticks)
plt.legend()
plt.savefig('cdpp_saturation.png')


