import halophot
from halophot.halo_tools import halo_tpf
import lightkurve 

def test_lk():
	print('testing')

	fname = "../data/ktwo200173843-c13_lpd-targ.fits" # aldebaran

	tpf = halo_tpf(fname)

	lc = tpf.to_lightcurve()
	lc.primary_header = tpf.hdu[0].header
	lc.data_header = tpf.hdu[1].header

	meta, corr_lc = tpf.halo(split_times=[3010],thresh=0.5);
	weightmap = meta['weightmap'][0]


def test_k2sc():
	import k2sc
	from k2sc import standalone

	tpf = halo_tpf(fname)

	lc = tpf.to_lightcurve()
	lc.primary_header = tpf.hdu[0].header
	lc.data_header = tpf.hdu[1].header

	meta, corr_lc = tpf.halo(split_times=[3010],thresh=0.5);


	corr_lc.__class__ = standalone.k2sc_lc

	corr_lc.k2sc()
