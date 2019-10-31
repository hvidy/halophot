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

'''--------------------------------------------------
halo_noisy.py - how does the noisiness of x y positions,
ie their correlations, affect the performance of halo
photometry? 
--------------------------------------------------'''

t = np.linspace(0,100,ncad)

amplitude = 1.

log_snrs = np.linspace(-3, 3,100)

x, y = amplitude*np.sin(2*np.pi*t/20.), amplitude*np.cos(2*np.pi*t/20.) # smooth


sigs_raw = 1.*np.zeros_like(log_snrs)
sigs_1 = 1.*np.zeros_like(log_snrs)

mads_raw = 1.*np.zeros_like(log_snrs)
mads_tv = 1.*np.zeros_like(log_snrs)

xamp = np.max(x)-np.min(x)
yamp = np.max(y)-np.min(y)

start = clock()

for jj, snr in enumerate(log_snrs):
	# print 'Doing period',period


	xx, yy = x + xamp*(10**snr)*np.random.randn(len(x)), y + yamp*(10**snr)*np.random.randn(len(y))
	xx = xx /(np.max(xx)-np.min(xx))*xamp
	yy = yy /(np.max(yy)-np.min(yy))*yamp


	f = 20*np.ones(ncad) + np.sin(t) # make this whatever function you like! 
	f[400:500] *= 0.9 # toy transit


	raw_lc, lc_opt_1 = do_sim(xx,yy,f)

	ss_raw = cdpp(t,raw_lc-f/np.nanmedian(f)+1)
	ss1 = cdpp(t,lc_opt_1-f/np.nanmedian(f)+1)

	mad_raw = mad(raw_lc,f/np.nanmedian(f))
	mad_tv = mad(lc_opt_1,f/np.nanmedian(f))

	sigs_raw[jj] = ss_raw 
	sigs_1[jj] = ss1

	mads_raw[jj] = mad_raw
	mads_tv[jj] = mad_tv


finish = clock()
print 'Done'
print 'Time elapsed:',finish-start

plt.figure(0)
plt.clf()
plt.plot(log_snrs,sigs_raw,'.-',label='Raw')
plt.plot(log_snrs,sigs_1,'.-',label='TV1')
plt.yscale('log')
plt.ylabel('CDPP (ppm)')
plt.xlabel('Log Dither Noise in XY variations')
plt.title(r'TV Performance from Smooth to Noisy Jitter')
plt.legend()
plt.savefig('snr_sweep.png')
plt.show()


plt.figure(1)
plt.clf()
plt.plot(log_snrs,mads_raw,'.-',label='Raw')
plt.plot(log_snrs,mads_tv,'.-',label='TV1')
plt.ylabel('MAD')
plt.xlabel('Log Dither Noise in XY variations')
plt.yscale('log')
plt.title(r'TV Performance from Smooth to Noisy Jitter')
plt.legend()
plt.savefig('snr_sweep_mad.png')
plt.show()