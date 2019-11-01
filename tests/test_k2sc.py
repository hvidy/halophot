import halophot
from halophot.halo_tools import halo_tpf
import lightkurve 

import os
TESTDIR = os.path.abspath(os.path.dirname(__file__))
ddir = os.path.join(TESTDIR,'../data/')


def test_k2sc():
	import k2sc
	from k2sc import standalone

	fname = ddir+"ktwo200128910-c111_lpd-targ.fits.gz" 

	tpf = halo_tpf(fname)

	lc = tpf.to_lightcurve()
	lc.primary_header = tpf.hdu[0].header
	lc.data_header = tpf.hdu[1].header

	meta, corr_lc = tpf.halo(split_times=[3010],thresh=0.5);

	corr_lc.__class__ = standalone.k2sc_lc

	corr_lc.k2sc()

def test_saturation():
	print('testing')

	fname = ddir+"ktwo200128910-c111_lpd-targ.fits.gz" 

	tpf = halo_tpf(fname)

	lc = tpf.to_lightcurve()
	lc.primary_header = tpf.hdu[0].header
	lc.data_header = tpf.hdu[1].header

	meta, corr_lc = tpf.halo(thresh=-1);
	weightmap = meta['weightmap'][0]
