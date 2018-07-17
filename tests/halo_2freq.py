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
halo_freq.py - how well does halo photometry work 
as a function of frequency?
--------------------------------------------------'''

'''------------------------
Generate a toy light curve
------------------------'''

t = np.linspace(0,100,ncad)

amplitude = 1.

xperiods = np.linspace(0.05,2.5,100)
fperiods = np.linspace(0.05,2.5,100)

xfreqs = np.linspace(1./2.5,1./0.05,200)
ffreqs = np.linspace(1./2.5,1./0.05,201)


nxp = np.size(xfreqs)
nfp = np.size(ffreqs)

cdpps = np.zeros((nxp,nfp))

mads = np.zeros((nxp,nfp))

start = clock()

for jj, xfreq in enumerate(xfreqs):
	for kk, ffreq in enumerate(ffreqs):
		xperiod, fperiod = 1./xfreq, 1./ffreq

		x, y = amplitude*np.sin(2*np.pi*t/xperiod), amplitude*np.cos(2*np.pi*t/xperiod) # smooth

		f = 20*np.ones(ncad) + np.sin(2*np.pi/fperiod*t) # make this whatever function you like! 

		raw_lc, lc_opt_1 = do_sim(x,y,f)

		ss1 = cdpp(t,lc_opt_1-f/np.nanmedian(f)+1)
		mad1 = mad(lc_opt_1,f/np.nanmedian(f))

		cdpps[jj,kk] = ss1 
		mads[jj,kk] = mad1 


finish = clock()
print 'Done'
print 'Time elapsed:',finish-start

plt.figure(0)
plt.clf()

plt.imshow(np.log10(cdpps),interpolation=None,extent=[1./2.5,1./0.05,1./2.5,1./0.05])
plt.colorbar(label="log CDPP")
plt.ylabel(r'xy variation Frequency ($d^{-1}$)')
plt.xlabel(r'Flux variation Frequency ($d^{-1}$)')
plt.title(r'TV Performance: Resonances')
plt.legend()
plt.savefig('freq_double.png')
plt.show()

plt.figure(1)
plt.clf()

plt.imshow(np.log10(mads),interpolation=None,extent=[1./2.5,1./0.05,1./2.5,1./0.05])
plt.colorbar(label="log MAD")
plt.ylabel(r'xy variation Frequency ($d^{-1}$)')
plt.xlabel(r'Flux variation Frequency ($d^{-1}$)')
plt.title(r'TV Performance: Resonances')
plt.legend()
plt.savefig('freq_double_mad.png')
plt.show()
