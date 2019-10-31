import numpy as np
from autograd import numpy as agnp
from autograd import grad 

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

'''--------------------------------------------------
halo_smooth.py - does halo photometry work with smooth
pointing variations or does it need jumpy ones?
--------------------------------------------------'''

'''------------------------
Generate a toy light curve
------------------------'''

t = np.linspace(0,100,ncad)

amplitude = 1.
period = 2.*np.pi
x, y = amplitude*np.sin(2*np.pi*t/period), amplitude*np.cos(2*np.pi*t/period)

np.random.seed(10)

x = 1.0*np.random.randn(x.shape[0])
y = 1.0*np.random.randn(y.shape[0])

f = 20*np.ones(ncad) + np.sin(t/6.) # make this whatever function you like! 
f[400:500] *= 0.990 # toy transit


'''------------------------
Define a PSF and aperture
------------------------'''

width = 3.
start = clock()

nx, ny = 10, 10
npix = nx*ny
pixels = np.zeros((nx,ny))

'''------------------------
Simulate data
------------------------'''

tpf = np.zeros((nx,ny,ncad))
sensitivity = 1-0.1*np.random.rand(nx,ny)
white = 0

for j in range(ncad):
    tpf[:,:,j] = f[j]*gaussian_psf(pixels,x[j],y[j],width)*sensitivity + np.random.randn(nx,ny)*white

pixelvector = np.reshape(tpf,(nx*ny,ncad))

'''------------------------
Define objectives
------------------------'''


def softmax(x):
    '''From https://gist.github.com/stober/1946926'''
    e_x = agnp.exp(x - agnp.max(x))
    out = e_x / e_x.sum()
    return out

def tv_soft(weights,lag):
    flux = agnp.dot(softmax(weights).T,pixelvector)
    diff = agnp.sum(agnp.abs(flux[lag:] - flux[:(-lag)]))
    return diff/agnp.mean(flux)

gradient = grad(tv_soft,argnum=0)

'''------------------------
Reconstruct lightcurves
------------------------'''

cons = ({'type': 'eq', 'fun': lambda z: z.sum() - 1.})
bounds = (npix)*((0,1),)

w_init = np.random.rand(npix)
w_init /= np.sum(w_init)
# w_init = np.ones(180)/180.

tic = clock()

lcs, cdpps = [], []
lags = np.arange(20).astype('int')+1

for lag in lags:
	# print(lag)
	res = optimize.minimize(tv_soft, w_init, args=(lag,),method='L-BFGS-B', jac=gradient, 
	            options={'disp': False,'maxiter':100})

	w_best = softmax(res['x'])
	toc = clock()

	# print('Time taken for TV1:',(toc-tic))

	lc_opt = np.dot(w_best.T,pixelvector)
	lcs.append(lc_opt)
	lc_opt /= np.nanmedian(lc_opt)
	ss1 = cdpp(t,lc_opt-f/np.nanmedian(f)+1)
	cdpps.append(ss1)

	print('%d lag CDPP (ppm):' % lag,ss1)



raw_lc = np.sum(pixelvector,axis=0)
raw_lc /= np.nanmedian(raw_lc)
ssr = cdpp(t,raw_lc-f/np.nanmedian(f)+1)

print('Raw Light Curve Noise (ppm):',ssr)


finish = clock()
print('Done')
print('Time elapsed:',finish-start)


plt.figure(0)
plt.clf()

plt.plot(t,raw_lc,'.',label='Raw')
for j,lag in enumerate(lags):
	plt.plot(t,lcs[j],'.',label=lag)

plt.plot(t,f/np.nanmedian(f),'-',label='True')
plt.ylabel('Time (d)')
plt.xlabel('Flux')
plt.title(r'%.1d Period : Light curves' % period)
plt.legend()
plt.savefig('period_%.0f_lc.png' % period)
plt.show()

plt.figure(1)
plt.clf()
plt.plot(lags,cdpps,'-')
plt.xlabel('Lag (cad)')
plt.ylabel('CDPP (ppm)')
plt.title('How long do you lag?')
plt.show()
