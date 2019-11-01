# halophot
![](https://github.com/hvidy/halophot/workflows/integration/badge.svg)
[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)
[![arXiv](http://img.shields.io/badge/arXiv-1708.07462-blue.svg?style=flat)](http://arxiv.org/abs/1708.07462)
[![arXiv](http://img.shields.io/badge/arXiv-1908.06981-blue.svg?style=flat)](http://arxiv.org/abs/1908.06981)
## Contributors

Tim White, Benjamin Pope

## Installation

Simply run

`pip install halophot`

or else clone this git repo, enter the directory, and run

`python setup.py install`

## K2 Halo Photometry

This is code that implements a Total Variation (TV) based regularization for Kepler/K2 photometry of very bright stars. 

We minimize nth order TV - i.e. the sum of the absolute values of the nth differences of a light curve - of a light curve created as the weighted sum of pixels, with weights in (0,1). This appears remarkably effective at removing pointing-based systematics from K2 lightcurves where it is impractical to do photometry otherwise and apply more standard detrending methods. 

We believe this is of practical use for Kepler targets brighter than  Kp ~ 6. 

## Basic usage

Call halophot either as a library of functions (/src/halo_tools.py) or from the command line as, e.g.

halo ktwo200007768-c04_lpd-targ.fits --data-dir /path/to/data/directory/ --name Atlas -c 4 --do-plot

where --name is the star name you would like to save the outputs under, and -c is the campaign.

## License

We invite anyone interested to use and modify this code under a GPL v3 license. 

## Citation

We request that anyone using this code for photometry of stars observed under the bright star GO programs in K2 include White and Pope as coauthors on any publications.

If you use our code, please cite

    White et al. (2017), MNRAS, 471, 2882-2901, arXiv:1708.07462 

and 

    Pope et al. (2019), arXiv: 1908.06981

Or use this these BibTeX entries:

    @ARTICLE{White2017,
       author = {{White}, T.~R. and {Pope}, B.~J.~S. and {Antoci}, V. and {P{\'a}pics}, P.~I. and 
      {Aerts}, C. and {Gies}, D.~R. and {Gordon}, K. and {Huber}, D. and 
      {Schaefer}, G.~H. and {Aigrain}, S. and {Albrecht}, S. and {Barclay}, T. and 
      {Barentsen}, G. and {Beck}, P.~G. and {Bedding}, T.~R. and {Fredslund Andersen}, M. and 
      {Grundahl}, F. and {Howell}, S.~B. and {Ireland}, M.~J. and 
      {Murphy}, S.~J. and {Nielsen}, M.~B. and {Silva Aguirre}, V. and 
      {Tuthill}, P.~G.},
        title = "{Beyond the Kepler/K2 bright limit: variability in the seven brightest members of the Pleiades}",
      journal = {\mnras},
    archivePrefix = "arXiv",
       eprint = {1708.07462},
     primaryClass = "astro-ph.SR",
     keywords = {asteroseismology, techniques: photometric, stars: early type, stars: variables: general, open clusters and associations: individual: Pleiades},
         year = 2017,
        month = nov,
       volume = 471,
        pages = {2882-2901},
          doi = {10.1093/mnras/stx1050},
       adsurl = {http://adsabs.harvard.edu/abs/2017MNRAS.471.2882W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    @ARTICLE{2019arXiv190806981P,
       author = {{Pope}, Benjamin J.~S. and {White}, Timothy R. and {Farr}, Will M. and
         {Yu}, Jie and {Greklek-McKeon}, Michael and {Huber}, Daniel and
         {Aerts}, Conny and {Aigrain}, Suzanne and {Bedding}, Timothy R. and
         {Boyajian}, Tabetha and {Creevey}, Orlagh L. and {Hogg}, David W.},
        title = "{The K2 Bright Star Survey I: Methodology and Data Release}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = "2019",
        month = "Aug",
          eid = {arXiv:1908.06981},
        pages = {arXiv:1908.06981},
    archivePrefix = {arXiv},
       eprint = {1908.06981},
     primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190806981P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }



