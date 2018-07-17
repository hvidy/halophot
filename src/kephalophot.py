
import pylab, numpy, pyfits
from pylab import *
from matplotlib import *
from numpy import *
from pyfits import *
import kepio, kepmsg, kepkey, kepplot, kepstat, kepfunc
import sys, time, re, math

# -----------------------------------------------------------
# core code

def kephalophot(infile,outfile,plotfile,plottype,filter,function,cutoff,clobber,verbose,logfile,status, cmdLine=False): 

# input arguments

    status = 0
    seterr(all="ignore") 

# log the call 

    hashline = '----------------------------------------------------------------------------'
    kepmsg.log(logfile,hashline,verbose)
    call = 'KEPHALOPHOT -- '
    call += 'infile='+infile+' '
    call += 'outfile='+outfile+' '
    call += 'plotfile='+plotfile+' '
    call += 'plottype='+plottype+' '
    filt = 'n'
    if (filter): filt = 'y'
    call += 'filter='+filt+ ' '
    call += 'function='+function+' '
    call += 'cutoff='+str(cutoff)+' '
    overwrite = 'n'
    if (clobber): overwrite = 'y'
    call += 'clobber='+overwrite+ ' '
    chatter = 'n'
    if (verbose): chatter = 'y'
    call += 'verbose='+chatter+' '
    call += 'logfile='+logfile
    kepmsg.log(logfile,call+'\n',verbose)

# start time

    kepmsg.clock('KEPHALOPHOT started at',logfile,verbose)

# test log file

    logfile = kepmsg.test(logfile)

# clobber output file

    if clobber: status = kepio.clobber(outfile,logfile,verbose)
    if kepio.fileexists(outfile): 
        message = 'ERROR -- KEPHALOPHOT: ' + outfile + ' exists. Use --clobber'
        status = kepmsg.err(logfile,message,verbose)

# open TPF FITS file

    if status == 0:
        kepid, channel, skygroup, module, output, quarter, season, \
            ra, dec, column, row, kepmag, xdim, ydim, barytime, status = \
            kepio.readTPF(infile,'TIME',logfile,verbose)
    if status == 0:
        kepid, channel, skygroup, module, output, quarter, season, \
            ra, dec, column, row, kepmag, xdim, ydim, tcorr, status = \
            kepio.readTPF(infile,'TIMECORR',logfile,verbose)
    if status == 0:
        kepid, channel, skygroup, module, output, quarter, season, \
            ra, dec, column, row, kepmag, xdim, ydim, cadno, status = \
            kepio.readTPF(infile,'CADENCENO',logfile,verbose)
    if status == 0:
        kepid, channel, skygroup, module, output, quarter, season, \
            ra, dec, column, row, kepmag, xdim, ydim, fluxpixels, status = \
            kepio.readTPF(infile,'FLUX',logfile,verbose)
    if status == 0:
        kepid, channel, skygroup, module, output, quarter, season, \
            ra, dec, column, row, kepmag, xdim, ydim, errpixels, status = \
            kepio.readTPF(infile,'FLUX_ERR',logfile,verbose)
    if status == 0:
        kepid, channel, skygroup, module, output, quarter, season, \
            ra, dec, column, row, kepmag, xdim, ydim, qual, status = \
            kepio.readTPF(infile,'QUALITY',logfile,verbose)

# read mask defintion data from TPF file

    if status == 0:
        maskimg, pixcoord1, pixcoord2, status = kepio.readMaskDefinition(infile,logfile,verbose)

# print target data

    if status == 0:
        print('')
        print('      KepID:  %s' % kepid)
        print(' RA (J2000):  %s' % ra)
        print('Dec (J2000): %s' % dec)
        print('     KepMag:  %s' % kepmag)
        print('   SkyGroup:    %2s' % skygroup)
        print('     Season:    %2s' % str(season))
        print('    Channel:    %2s' % channel)
        print('     Module:    %2s' % module)
        print('     Output:     %1s' % output)
        print('')

# how many quality = 0 rows? how many pixels?

    if status == 0:
        np = ydim*xdim
        nrows = len(fluxpixels)
        npts = 0
        for i in range(nrows):
            if qual[i] < 1e4 and \
                    numpy.isfinite(barytime[i]) and \
                    numpy.isfinite(fluxpixels[i,ydim*xdim/2]):
                npts += 1
        time = empty((npts))
        timecorr = empty((npts))
        cadenceno = empty((npts))
        quality = empty((npts))
        pixseries = zeros((npts,np))
        errseries = zeros((npts,np))
        # pixseries = empty((ydim,xdim,npts))
        # errseries = empty((ydim,xdim,npts))

# construct output light curves

    if status == 0:
        for i in range(np):
            npts = 0
            for j in range(nrows):
                if qual[j] < 1e4 and \
                numpy.isfinite(barytime[j]) and \
                numpy.isfinite(fluxpixels[j,i]):
                    time[npts] = barytime[j]
                    timecorr[npts] = tcorr[j]
                    cadenceno[npts] = cadno[j]
                    quality[npts] = qual[j]
                    pixseries[npts,i] = fluxpixels[j,i]
                    errseries[npts,i] = errpixels[j,i]
                    npts += 1

# define data sampling

    if status == 0 and filter:
        tpf, status = kepio.openfits(infile,'readonly',logfile,verbose)
    if status == 0 and filter:
        cadence, status = kepkey.cadence(tpf[1],infile,logfile,verbose)     
        tr = 1.0 / (cadence / 86400)
        timescale = 1.0 / (cutoff / tr)

# define convolution function

    if status == 0 and filter:
        if function == 'boxcar':
            filtfunc = numpy.ones(numpy.ceil(timescale))
        elif function == 'gauss':
            timescale /= 2
            dx = numpy.ceil(timescale * 10 + 1)
            filtfunc = kepfunc.gauss()
            filtfunc = filtfunc([1.0,dx/2-1.0,timescale],linspace(0,dx-1,dx))
        elif function == 'sinc':
            dx = numpy.ceil(timescale * 12 + 1)
            fx = linspace(0,dx-1,dx)
            fx = fx - dx / 2 + 0.5
            fx /= timescale
            filtfunc = numpy.sinc(fx)
        filtfunc /= numpy.sum(filtfunc)

# pad time series at both ends with noise model

    if status == 0 and filter:
        for i in range(ydim):
            for j in range(xdim):
                ave, sigma  = kepstat.stdev(pixseries[i,j,:len(filtfunc)])
                padded = numpy.append(kepstat.randarray(numpy.ones(len(filtfunc)) * ave, \
                                                            numpy.ones(len(filtfunc)) * sigma), pixseries[i,j,:])
                ave, sigma  = kepstat.stdev(pixseries[i,j,-len(filtfunc):])
                padded = numpy.append(padded, kepstat.randarray(numpy.ones(len(filtfunc)) * ave, \
                                                                    numpy.ones(len(filtfunc)) * sigma))

# convolve data

                if status == 0:
                    convolved = convolve(padded,filtfunc,'same')

# remove padding from the output array

                if status == 0:
                    outdata = convolved[len(filtfunc):-len(filtfunc)]
            
# subtract low frequencies

                if status == 0:
                    outmedian = median(outdata)
                    pixseries[i,j,:] = pixseries[i,j,:] - outdata + outmedian

# construct weighted time series
    if status == 0:
        wgt = numpy.ones((np,3))
        twgt = numpy.ones((np,3))
        wgt /= sum(wgt,axis=0)
        satlvl = 0.8*numpy.max(numpy.max(pixseries,axis=1))
        brk1 = 9.7257203
        brk2 = 45.
        ind1 = where(time-time[0] < brk1)
        ind2 = where((time-time[0] >= brk1) & (time-time[0] < brk2 ))
        ind3 = where(time-time[0] >= brk2)
        z=numpy.array([0.0,0.0,0.0])
        for i in range(np):
            if max(pixseries[ind1,i].flatten()) > satlvl or max(pixseries[ind1,i].flatten()) <= 100: wgt[i,0]=0; z[0]+=1
            if max(pixseries[ind2,i].flatten()) > satlvl or max(pixseries[ind2,i].flatten()) <= 100: wgt[i,1]=0; z[1]+=1
            if max(pixseries[ind3,i].flatten()) > satlvl or max(pixseries[ind3,i].flatten()) <= 100: wgt[i,2]=0; z[2]+=1
        print(z)
        print(np-z)
        sf1 = numpy.dot(pixseries[ind1,:],wgt[:,0]).flatten()
        sf2 = numpy.dot(pixseries[ind2,:],wgt[:,1]).flatten()
        sf3 = numpy.dot(pixseries[ind3,:],wgt[:,2]).flatten()
        sf1 /= numpy.median(sf1)
        sf2 /= numpy.median(sf2)
        sf3 /= numpy.median(sf3)
        originalflux = numpy.concatenate([sf1,sf2,sf3])

#        a=numpy.array([0.0,0.0,0.0])        
#        t=0
#        ca = numpy.array([0.0,0.0,0.0])
#        ct = 0
#        sig1 = numpy.std(sf1)
#        sig2 = numpy.std(sf2)
#        sig3 = numpy.std(sf3)
#        while 1:
#            j = int(numpy.floor(numpy.random.random()*np))
#            if sum(wgt[j,:]) == 0: continue
#            if ct == 1000:
#                print(ca)
#                if ca[0] < 333 and ca[1] < 333 and ca[2] < 333: break
#                ca = numpy.array([0.0,0.0,0.0])
#                ct = 0
#            t += 1
#            ct += 1
#            wgt /= sum(wgt,axis=0)
#            twgt=copy(wgt)
#            twgt[j,:]*=numpy.random.normal(1.0,0.05,3)
#            twgt /= sum(twgt,axis=0)
#            tsf1 = numpy.dot(pixseries[ind1,:],twgt[:,0]).flatten()
#            tsf2 = numpy.dot(pixseries[ind2,:],twgt[:,1]).flatten()
#            tsf3 = numpy.dot(pixseries[ind3,:],twgt[:,2]).flatten()
#            tsf1 /= numpy.median(tsf1)
#            tsf2 /= numpy.median(tsf2)
#            tsf3 /= numpy.median(tsf3)
#            tsig1 = numpy.std(tsf1)
#            tsig2 = numpy.std(tsf2)
#            tsig3 = numpy.std(tsf3)
#            if tsig1 < sig1:
#                wgt[:,0] = twgt[:,0]
#                sig1 = tsig1
#                a[0] += 1
#                ca[0] += 1
#            if tsig2 < sig2:
#                wgt[:,1] = twgt[:,1]
#                sig2 = tsig2
#                a[1] += 1
#                ca[1] += 1
#            if tsig3 < sig3:
#                wgt[:,2] = twgt[:,2]
#                sig3 = tsig3
#                a[2] += 1
#                ca[2] += 1
#        print(100*a/t)
#        sf1 = numpy.dot(pixseries[ind1,:],wgt[:,0]).flatten()
#        sf2 = numpy.dot(pixseries[ind2,:],wgt[:,1]).flatten()
#        sf3 = numpy.dot(pixseries[ind3,:],wgt[:,2]).flatten()
#        sf1 /= numpy.median(sf1)
#        sf2 /= numpy.median(sf2)
#        sf3 /= numpy.median(sf3)
#
#        a=numpy.array([0.0,0.0,0.0])
#        t=0
#        ca = numpy.array([0.0,0.0,0.0])
#        ct = 0
#        sig1 = sum(numpy.fabs(sf1[1:]-sf1[:-1]))
#        sig2 = sum(numpy.fabs(sf2[1:]-sf2[:-1]))
#        sig3 = sum(numpy.fabs(sf3[1:]-sf3[:-1]))
#        while 1:
#            j = int(numpy.floor(numpy.random.random()*np))
#            if sum(wgt[j,:]) == 0: continue
#            if ct == 1000:
#                print(ca)
#                if ca[0] < 167 and ca[1] < 167 and ca[2] < 167: break#
#                ca = numpy.array([0.0,0.0,0.0])
#                ct = 0
#            t += 1
#            ct += 1
#            wgt /= sum(wgt,axis=0)
#            twgt=copy(wgt)
#            twgt[j,:]*=numpy.random.normal(1.0,0.05,3)
#            twgt /= sum(twgt,axis=0)
#            tsf1 = numpy.dot(pixseries[ind1,:],twgt[:,0]).flatten()
#            tsf2 = numpy.dot(pixseries[ind2,:],twgt[:,1]).flatten()
#            tsf3 = numpy.dot(pixseries[ind3,:],twgt[:,2]).flatten()
#            tsf1 /= numpy.median(tsf1)
#            tsf2 /= numpy.median(tsf2)
#            tsf3 /= numpy.median(tsf3)
#            tsig1 = sum(numpy.fabs(tsf1[1:]-tsf1[:-1]))
#            tsig2 = sum(numpy.fabs(tsf2[1:]-tsf2[:-1]))
#            tsig3 = sum(numpy.fabs(tsf3[1:]-tsf3[:-1]))
#            if tsig1 < sig1:
#                wgt[:,0] = twgt[:,0]
#                sig1 = tsig1
#                a[0] += 1
#                ca[0] += 1
#            if tsig2 < sig2:
#                wgt[:,1] = twgt[:,1]
#                sig2 = tsig2
#                a[1] += 1
#                ca[1] += 1
#            if tsig3 < sig3:
#                wgt[:,2] = twgt[:,2]
#                sig3 = tsig3
#                a[2] += 1
#                ca[2] += 1
#        print(100*a/t)
#        sf1 = numpy.dot(pixseries[ind1,:],wgt[:,0]).flatten()
#        sf2 = numpy.dot(pixseries[ind2,:],wgt[:,1]).flatten()
#        sf3 = numpy.dot(pixseries[ind3,:],wgt[:,2]).flatten()
#        sf1 /= numpy.median(sf1)
#        sf2 /= numpy.median(sf2)
#        sf3 /= numpy.median(sf3)

        a=numpy.array([0.0,0.0,0.0])
        t=0
        ca = numpy.array([0.0,0.0,0.0])
        ct = 0
        sig1 = sum(numpy.fabs(sf1[2:]-2*sf1[1:-1]+sf1[:-2]))
        sig2 = sum(numpy.fabs(sf2[2:]-2*sf2[1:-1]+sf2[:-2]))
        sig3 = sum(numpy.fabs(sf3[2:]-2*sf3[1:-1]+sf3[:-2]))
        while 1:
            j = int(numpy.floor(numpy.random.random()*np))
            if sum(wgt[j,:]) == 0: continue
            if ct == 1000:
                print(ca)
                if ca[0] < 20 and ca[1] < 20 and ca[2] < 20: break
                if t > 1000000: break
                ca = numpy.array([0.0,0.0,0.0])
                ct = 0
            t += 1
            ct += 1
            wgt /= sum(wgt,axis=0)
            twgt=copy(wgt)
            twgt[j,:]*=numpy.random.normal(1.0,0.05,3)
            twgt /= sum(twgt,axis=0)
            tsf1 = numpy.dot(pixseries[ind1,:],twgt[:,0]).flatten()
            tsf2 = numpy.dot(pixseries[ind2,:],twgt[:,1]).flatten()
            tsf3 = numpy.dot(pixseries[ind3,:],twgt[:,2]).flatten()
            tsf1 /= numpy.median(tsf1)
            tsf2 /= numpy.median(tsf2)
            tsf3 /= numpy.median(tsf3)
            tsig1 = sum(numpy.fabs(tsf1[2:]-2*tsf1[1:-1]+tsf1[:-2]))
            tsig2 = sum(numpy.fabs(tsf2[2:]-2*tsf2[1:-1]+tsf2[:-2]))
            tsig3 = sum(numpy.fabs(tsf3[2:]-2*tsf3[1:-1]+tsf3[:-2]))
            if tsig1 < sig1:
                wgt[:,0] = twgt[:,0]
                sig1 = tsig1
                a[0] += 1
                ca[0] += 1
            if tsig2 < sig2:
                wgt[:,1] = twgt[:,1]
                sig2 = tsig2
                a[1] += 1
                ca[1] += 1
            if tsig3 < sig3:
                wgt[:,2] = twgt[:,2]
                sig3 = tsig3
                a[2] += 1
                ca[2] += 1
        print(100*a/t)
        sf1 = numpy.dot(pixseries[ind1,:],wgt[:,0]).flatten()
        sf2 = numpy.dot(pixseries[ind2,:],wgt[:,1]).flatten()
        sf3 = numpy.dot(pixseries[ind3,:],wgt[:,2]).flatten()
        sf1 /= numpy.median(sf1)
        sf2 /= numpy.median(sf2)
        sf3 /= numpy.median(sf3)

        finalflux = numpy.concatenate([sf1,sf2,sf3])

# construct output file

    if status == 0:
        instruct, status = kepio.openfits(infile,'readonly',logfile,verbose)
        status = kepkey.history(call,instruct[0],outfile,logfile,verbose)
        hdulist = HDUList(instruct[0])
        cols = []
        cols.append(Column(name='TIME',format='D',unit='BJD - 2454833',disp='D12.7',array=time))
        cols.append(Column(name='TIMECORR',format='E',unit='d',disp='E13.6',array=timecorr))
        cols.append(Column(name='CADENCENO',format='J',disp='I10',array=cadenceno))
        cols.append(Column(name='QUALITY',format='J',array=quality))
        cols.append(Column(name='ORGFLUX',format='E',disp='E13.6',array=originalflux))
        cols.append(Column(name='FLUX',format='E',disp='E13.6',array=finalflux))
        # for i in range(ydim):
        #     for j in range(xdim):
        #         colname = 'COL%d_ROW%d' % (i+column,j+row)
        #         cols.append(Column(name=colname,format='E',disp='E13.6',array=pixseries[i,j,:]))
        hdu1 = new_table(ColDefs(cols))
        try:
            hdu1.header.update('INHERIT',True,'inherit the primary header')
        except:
            status = 0
        try:
            hdu1.header.update('EXTNAME','PIXELSERIES','name of extension')
        except:
            status = 0
        try:
            hdu1.header.update('EXTVER',instruct[1].header['EXTVER'],'extension version number (not format version)')
        except:
            status = 0
        try:
            hdu1.header.update('TELESCOP',instruct[1].header['TELESCOP'],'telescope')
        except:
            status = 0
        try:
            hdu1.header.update('INSTRUME',instruct[1].header['INSTRUME'],'detector type')
        except:
            status = 0
        try:
            hdu1.header.update('OBJECT',instruct[1].header['OBJECT'],'string version of KEPLERID')
        except:
            status = 0
        try:
            hdu1.header.update('KEPLERID',instruct[1].header['KEPLERID'],'unique Kepler target identifier')
        except:
            status = 0
        try:
            hdu1.header.update('RADESYS',instruct[1].header['RADESYS'],'reference frame of celestial coordinates')
        except:
            status = 0
        try:
            hdu1.header.update('RA_OBJ',instruct[1].header['RA_OBJ'],'[deg] right ascension from KIC')
        except:
            status = 0
        try:
            hdu1.header.update('DEC_OBJ',instruct[1].header['DEC_OBJ'],'[deg] declination from KIC')
        except:
            status = 0
        try:
            hdu1.header.update('EQUINOX',instruct[1].header['EQUINOX'],'equinox of celestial coordinate system')
        except:
            status = 0
        try:
            hdu1.header.update('TIMEREF',instruct[1].header['TIMEREF'],'barycentric correction applied to times')
        except:
            status = 0
        try:
            hdu1.header.update('TASSIGN',instruct[1].header['TASSIGN'],'where time is assigned')
        except:
            status = 0
        try:
            hdu1.header.update('TIMESYS',instruct[1].header['TIMESYS'],'time system is barycentric JD')
        except:
            status = 0
        try:
            hdu1.header.update('BJDREFI',instruct[1].header['BJDREFI'],'integer part of BJD reference date')
        except:
            status = 0
        try:
            hdu1.header.update('BJDREFF',instruct[1].header['BJDREFF'],'fraction of the day in BJD reference date')
        except:
            status = 0
        try:
            hdu1.header.update('TIMEUNIT',instruct[1].header['TIMEUNIT'],'time unit for TIME, TSTART and TSTOP')
        except:
            status = 0
        try:
            hdu1.header.update('TSTART',instruct[1].header['TSTART'],'observation start time in BJD-BJDREF')
        except:
            status = 0
        try:
            hdu1.header.update('TSTOP',instruct[1].header['TSTOP'],'observation stop time in BJD-BJDREF')
        except:
            status = 0
        try:
            hdu1.header.update('LC_START',instruct[1].header['LC_START'],'mid point of first cadence in MJD')
        except:
            status = 0
        try:
            hdu1.header.update('LC_END',instruct[1].header['LC_END'],'mid point of last cadence in MJD')
        except:
            status = 0
        try:
            hdu1.header.update('TELAPSE',instruct[1].header['TELAPSE'],'[d] TSTOP - TSTART')
        except:
            status = 0
        try:
            hdu1.header.update('LIVETIME',instruct[1].header['LIVETIME'],'[d] TELAPSE multiplied by DEADC')
        except:
            status = 0
        try:
            hdu1.header.update('EXPOSURE',instruct[1].header['EXPOSURE'],'[d] time on source')
        except:
            status = 0
        try:
            hdu1.header.update('DEADC',instruct[1].header['DEADC'],'deadtime correction')
        except:
            status = 0
        try:
            hdu1.header.update('TIMEPIXR',instruct[1].header['TIMEPIXR'],'bin time beginning=0 middle=0.5 end=1')
        except:
            status = 0
        try:
            hdu1.header.update('TIERRELA',instruct[1].header['TIERRELA'],'[d] relative time error')
        except:
            status = 0
        try:
            hdu1.header.update('TIERABSO',instruct[1].header['TIERABSO'],'[d] absolute time error')
        except:
            status = 0
        try:
            hdu1.header.update('INT_TIME',instruct[1].header['INT_TIME'],'[s] photon accumulation time per frame')
        except:
            status = 0
        try:
            hdu1.header.update('READTIME',instruct[1].header['READTIME'],'[s] readout time per frame')
        except:
            status = 0
        try:
            hdu1.header.update('FRAMETIM',instruct[1].header['FRAMETIM'],'[s] frame time (INT_TIME + READTIME)')
        except:
            status = 0
        try:
            hdu1.header.update('NUM_FRM',instruct[1].header['NUM_FRM'],'number of frames per time stamp')
        except:
            status = 0
        try:
            hdu1.header.update('TIMEDEL',instruct[1].header['TIMEDEL'],'[d] time resolution of data')
        except:
            status = 0
        try:
            hdu1.header.update('DATE-OBS',instruct[1].header['DATE-OBS'],'TSTART as UTC calendar date')
        except:
            status = 0
        try:
            hdu1.header.update('DATE-END',instruct[1].header['DATE-END'],'TSTOP as UTC calendar date')
        except:
            status = 0
        try:
            hdu1.header.update('BACKAPP',instruct[1].header['BACKAPP'],'background is subtracted')
        except:
            status = 0
        try:
            hdu1.header.update('DEADAPP',instruct[1].header['DEADAPP'],'deadtime applied')
        except:
            status = 0
        try:
            hdu1.header.update('VIGNAPP',instruct[1].header['VIGNAPP'],'vignetting or collimator correction applied')
        except:
            status = 0
        try:
            hdu1.header.update('GAIN',instruct[1].header['GAIN'],'[electrons/count] channel gain')
        except:
            status = 0
        try:
            hdu1.header.update('READNOIS',instruct[1].header['READNOIS'],'[electrons] read noise')
        except:
            status = 0
        try:
            hdu1.header.update('NREADOUT',instruct[1].header['NREADOUT'],'number of read per cadence')
        except:
            status = 0
        try:
            hdu1.header.update('TIMSLICE',instruct[1].header['TIMSLICE'],'time-slice readout sequence section')
        except:
            status = 0
        try:
            hdu1.header.update('MEANBLCK',instruct[1].header['MEANBLCK'],'[count] FSW mean black level')
        except:
            status = 0
        hdulist.append(hdu1)
        hdulist.writeto(outfile)
        status = kepkey.new('EXTNAME','APERTURE','name of extension',instruct[2],outfile,logfile,verbose)
        pyfits.append(outfile,instruct[2].data,instruct[2].header)
        wgt1=numpy.reshape(wgt[:,0],(ydim,xdim))
        wgt2=numpy.reshape(wgt[:,1],(ydim,xdim))        
        wgt3=numpy.reshape(wgt[:,2],(ydim,xdim))
        hdu3 = ImageHDU(data=wgt1,header=instruct[2].header,name='WEIGHTS1')
        hdu4 = ImageHDU(data=wgt2,header=instruct[2].header,name='WEIGHTS2')
        hdu5 = ImageHDU(data=wgt3,header=instruct[2].header,name='WEIGHTS3')
        pyfits.append(outfile,hdu3.data,hdu3.header)
        pyfits.append(outfile,hdu4.data,hdu4.header)
        pyfits.append(outfile,hdu5.data,hdu5.header)
        status = kepio.closefits(instruct,logfile,verbose)
    else:
        message = 'WARNING -- KEPHALOPHOT: output FITS file requires > 999 columns. Non-compliant with FITS convention.'

        kepmsg.warn(logfile,message)

# plot style

    if status == 0:
        try:
            params = {'backend': 'png',
                      'axes.linewidth': 2.0,
                      'axes.labelsize': 32,
                      'axes.font': 'sans-serif',
                      'axes.fontweight' : 'bold',
                      'text.fontsize': 8,
                      'legend.fontsize': 8,
                      'xtick.labelsize': 12,
                      'ytick.labelsize': 12}
            pylab.rcParams.update(params)
        except:
            pass

# plot pixel array

    fmin = 1.0e33
    fmax = -1.033
    if status == 0:
        pylab.figure(num=None,figsize=[12,12])
        pylab.clf()
        dx = 0.93 #/ xdim
        dy = 0.94 #/ ydim
        ax = pylab.axes([0.06,0.05,0.93,0.94])
        pylab.gca().xaxis.set_major_formatter(pylab.ScalarFormatter(useOffset=False))
        pylab.gca().yaxis.set_major_formatter(pylab.ScalarFormatter(useOffset=False))
        pylab.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        pylab.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        labels = ax.get_yticklabels()
        setp(labels, 'rotation', 90, fontsize=12)
        pylab.xlim(numpy.min(pixcoord1) - 0.5,numpy.max(pixcoord1) + 0.5)
        pylab.ylim(numpy.min(pixcoord2) - 0.5,numpy.max(pixcoord2) + 0.5)
        pylab.xlabel('time', {'color' : 'k'})
        pylab.ylabel('arbitrary flux', {'color' : 'k'})
        tmin = amin(time)
        tmax = amax(time)
        try:
            numpy.isfinite(amin(finalflux))
            numpy.isfinite(amin(finalflux))
            fmin = amin(finalflux)
            fmax = amax(finalflux)
        except:
            ugh = 1
        xmin = tmin - (tmax - tmin) / 40
        xmax = tmax + (tmax - tmin) / 40
        ymin = fmin - (fmax - fmin) / 20
        ymax = fmax + (fmax - fmin) / 20
        pylab.axes([0.06,0.05,dx,dy])
        pylab.setp(pylab.gca(),xticklabels=[],yticklabels=[])
        ptime = time * 1.0
        ptime = numpy.insert(ptime,[0],ptime[0])
        ptime = numpy.append(ptime,ptime[-1])
        pflux = finalflux * 1.0
        pflux = numpy.insert(pflux,[0],-1000.0)
        pflux = numpy.append(pflux,-1000.0)
        pylab.plot(time,finalflux,color='#0000ff',linestyle='-',linewidth=0.5)
        pylab.fill(ptime,pflux,fc='#FFF380',linewidth=0.0,alpha=1.0)
        if 'loc' in plottype:
            pylab.xlim(xmin,xmax)
            pylab.ylim(ymin,ymax)
        if 'glob' in plottype:
            pylab.xlim(xmin,xmax)
            pylab.ylim(1.0e-10,numpy.nanmax(pixseries) * 1.05)
        if 'full' in plottype:
            pylab.xlim(xmin,xmax)
            pylab.ylim(1.0e-10,ymax * 1.05)

# render plot

        if cmdLine: 
            pylab.show()
        else: 
            pylab.ion()
            pylab.plot([])
            pylab.ioff()    
        if plotfile.lower() != 'none':
            pylab.savefig(plotfile)

# stop time

    if status == 0:
        kepmsg.clock('KEPHALOPHOT ended at',logfile,verbose)

    return

# -----------------------------------------------------------
# main
if '--shell' in sys.argv:
    import argparse
    
    parser = argparse.ArgumentParser(description='Individual time series photometry for all pixels within a target mask')
    parser.add_argument('--shell', action='store_true', help='Are we running from the shell?')

    parser.add_argument('infile', help='Name of input file', type=str)

    parser.add_argument('outfile', help='Name of FITS file to output', type=str)
    parser.add_argument('--plotfile', default='None', help='name of output PNG plot file', type=str)

    parser.add_argument('--plottype', default='global', help='Plotting type', type=str, choices=['local','global','full'])

    parser.add_argument('--filter', action='store_true', help='High-pass Filter data?')
    parser.add_argument('--function', default='boxcar', help='Type of filter', type=str, choices=['boxcar','gauss','sinc'])
    parser.add_argument('--cutoff', default=1.0, help='Characteristic frequency cutoff of filter [1/days]', type=float)
    

    parser.add_argument('--clobber', action='store_true', help='Overwrite output file?')
    parser.add_argument('--verbose', action='store_true', help='Write to a log file?')
    parser.add_argument('--logfile', '-l', help='Name of ascii log file', default='kepcotrend.log', dest='logfile', type=str)
    parser.add_argument('--status', '-e', help='Exit status (0=good)', default=0, dest='status', type=int)


    args = parser.parse_args()
    
    cmdLine=True

    kephalophot(args.infile,args.outfile,args.plotfile,args.plottype,
        args.filter,args.function,args.cutoff,args.clobber,args.verbose,args.logfile,args.status, cmdLine)
    

else:
    from pyraf import iraf
    parfile = iraf.osfn("kepler$kephalophot.par")
    t = iraf.IrafTaskFactory(taskname="kephalophot", value=parfile, function=kephalophot)
