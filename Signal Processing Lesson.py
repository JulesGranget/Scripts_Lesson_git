

import os

import numpy as np

import matplotlib.pyplot as plt

import scipy.io # for loading data
import scipy.signal
import scipy.fftpack

import pandas as pd

from mpl_toolkits import mplot3d # for 3D plot

import mne







########################################
######## GENERATE FUNCTIONS ########
########################################

    #### Sinewave
srate = 1000
time = np.arange(-1,1+1/srate,1/srate) #This must have odd number of points to have the morlet wavelets centered
freq = 4
ampl = 2
phase = np.pi / 2
sinewave  = ampl * np.sin(2*np.pi*freq*time + phase) # cosine wave used to have a centered wavelet

plt.plot(time,sinewave), plt.show()

    #### Manual Gaussian window FWHM
srate = 1000
fwhm = 25 # in ms
time =  np.arange(-1,1+1/srate,1/srate) #This must have odd number of points to be centered
gausw = np.exp( -(4*np.log(2)*(time**2)) / fwhm**2 )

plt.plot(time,gausw), plt.show()

    #### Manual Gaussian window SIGMA
srate = 1000
time =  np.arange(-1,1+1/srate,1/srate)
ncycle = 7 # number of cycle that i want in my wavelet
s = ncycle / (2*np.pi*freq) 
gw = np.exp(-time**2/ (2*s**2)) 

plt.plot(time,gw), plt.show()

    #### Python Gaussian window
srate = 1000
nwind = int( 2*srate+1 ) # window length in seconds*srate
time =  np.arange(-1,1+1/srate,1/srate)
std = 100
win = scipy.signal.windows.gaussian(nwind,std)

plt.plot(time,win), plt.show()

    #### Manual Hann window
srate = 1000
nwind = int( 2*srate+1 )

hannw = .5 - np.cos(2*np.pi*np.linspace(0,1,int(nwind)))/2 # hann window

plt.plot(hannw), plt.show()

    #### Python Hann window
srate = 1000
nwind = int( 2*srate+1) # window length in seconds*srate

win = scipy.signal.windows.hann(nwind)

plt.plot(win), plt.show()

    #### Morlet wavelets

srate = 1000
time = np.arange(-1,1+1/srate,1/srate) 
freq = 30
ampl = 1
csw = ampl*np.exp(1j*(2*np.pi*freq*time)) # generate a complex sinewave

ncycle = 7
s = ncycle / (2*np.pi*freq) # number of cycle that i want in my wavelet
gw = np.exp(-time**2/ (2*s**2)) 
cmw = csw * gw # generate a complex morlets wavelet

# real part plot
fig, ax = plt.subplots()
plt.plot(time, np.real(cmw))
ax.set_xlabel('Time')
ax.set_ylabel('Real Part')
plt.show()

# 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(time, np.real(cmw), np.imag(cmw))
ax.set_xlabel('Time')
ax.set_ylabel('Real Part')
ax.set_zlabel('Img Part')
ax.legend()
plt.show()

    #### Morlet wavelets Family

# wavelet parameters
srate = 1000
wavetime = np.arange(-2,2,1/srate) 
nfrex = 50 # 50 frequencies
frex  = np.linspace(5,70,nfrex)
wavelets = np.zeros( (nfrex,len(wavetime)) ,dtype=complex) # matrix for all wavelets
ncycle = 7 # number of cycle in each wavelets

# create complex Morlet wavelet family
for fi in range(0,nfrex):
    # complex Morlet wavelet
    s = ncycle / (2*np.pi*frex[fi]) # number of cycle that i want in my wavelet
    gw = np.exp(-wavetime**2/ (2*s**2)) 
    sw = np.exp(1j*2*np.pi*frex[fi]*wavetime)
    mw =  gw * sw
    # fill the matrix
    wavelets[fi,:] = mw
    
# plot all the wavelets
plt.pcolormesh(wavetime,frex,np.real(wavelets))
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.title('Real part of wavelets')
plt.show()

    #### Morlet wavelets Family Adjusted

# wavelet parameters
srate = 1000
wavetime = np.arange(-2,2,1/srate) 
nfrex = 50 # 50 frequencies
frex  = np.linspace(5,70,nfrex)
wavelets = np.zeros( (nfrex,len(wavetime)) ,dtype=complex) # matrix for all wavelets
ncycle_list = np.linspace(7,20,nfrex)

# create complex Morlet wavelet family
for fi in range(nfrex):
    # complex Morlet wavelet
    s = ncycle_list[fi] / (2*np.pi*frex[fi]) # number of cycle that i want in my wavelet
    gw = np.exp(-wavetime**2/ (2*s**2)) 
    sw = np.exp(1j*2*np.pi*frex[fi]*wavetime)
    mw =  gw * sw
    # fill the matrix
    wavelets[fi,:] = mw
    
# plot all the wavelets
plt.pcolormesh(wavetime,frex,np.real(wavelets))
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.title('Real part of wavelets')
plt.show()



    #### Chirp

srate = 1000
time = np.arange(-4,4+1/srate,1/srate)

#chirp parameters
minfreq = 5
maxfreq = 17

#laplace distribution
fm = np.exp(-time**2) #fm for frequency modulation
fm -= np.min(fm)
fm /= np.max(fm)
fm = fm*(maxfreq-minfreq) + minfreq

#plt.plot(time,fm), plt.show()

chirp = np.sin(2*np.pi*(time+np.cumsum(fm)/srate))

plt.plot(time,chirp), plt.show()















################################
######## CREATE SIGNAL ########
################################

    #### Generate random noisy signal

srate = 1000 # Hz
n = 3*srate
time = np.arange(0,n,1/srate)
p = 15 # point to generate signal

time_interpol = np.arange(0,p) # times for global signal
amp_interpol = np.random.rand(p)*30 # amp for interpol sig
vec_interpol = np.linspace(0,p,n) #vec to interpolate
ampl   = np.interp(vec_interpol,time_interpol,amp_interpol)

noiseamp = 5
noise  = noiseamp * np.random.randn(n)

sig = ampl + noise



    #### Generate a multispectral noisy signal

# simulation parameters
srate = 1000 # in Hz
n = srate*256 # in seconds
time  = np.arange(0,n)/srate # timevec
frex  = [ 10,20,50 ] # frequencies to include

sig = np.zeros(len(time)) # initiate signal vec

# signal generation
for fi in range(0,len(frex)):
    sig = sig + np.sin(2*np.pi*frex[fi]*time)

sig = sig + np.random.randn(len(sig)) # add some noise

# signal inspection
plt.plot(time,sig)
plt.show()

hzPxx = np.linspace(0,srate/2,int(np.floor(n/2)))
Pxx = np.abs(scipy.fftpack.fft(sig,n)/n)**2
plt.plot(hzPxx,Pxx[0:len(hzPxx)],'k')
plt.xlabel('Frequency (norm.)')
plt.ylabel('Energy')
plt.title('Frequency-domain signal representation')
plt.show()


    #### Generate signal with ifft

srate  = 1000
N  = srate*2 # in sec
 
fwhm = .1
shift = .2
hz = 2
gwin = 1000*np.exp( -(4*np.log(2)*(hz-shift)/.1)**2 ) # gaussian window
fc = gwin * np.exp(2*np.pi*1j*np.random.rand(N)) # fourier coeff
data = np.real(scipy.fftpack.ifft( fc )) # ifft from fc
data = data + np.random.randn(N) # add noise

plt.plot(gwin)
plt.show()

plt.plot(data,'k')
plt.title('Original signal')
plt.xlabel('Time (a.u.)')
plt.show()

plt.plot(hz,np.abs(scipy.fftpack.fft(data))**2,'k')
plt.xlim([0,.5])
plt.xlabel('Frequency (norm.)')
plt.ylabel('Energy')
plt.title('Frequency-domain signal representation')
plt.show()




    #### Generate more biological signal with pink noise

# generate 1/f noise
srate  = 1000
N   = 8*srate
width = 200
pink_n = np.random.rand(N) * np.exp(-np.arange(0,N)/width) # generate 1/f pink noise with inverse (negative) exponential
fc  = pink_n * np.exp(2*np.pi*1j*np.random.rand(len(pink_n))) # generate fourier coeff based on noise, we add imaginary part 
amp = 8000 # amplitude for noise
noise = amp * np.real(scipy.fftpack.ifft(fc)) # take the ifft to go in time domain 

hzPxx = np.linspace(0, srate/2, int(np.floor(N/2)+1))
Pxx = np.abs(scipy.fftpack.fft(noise,N)/N)**2
Pxx = Pxx[0:len(hzPxx)] # taking only positive frequencies

plt.plot(noise,'k')
plt.title('Pink Noise Time Domain')
plt.xlabel('Time (a.u.)')
plt.show()

plt.plot(hzPxx, Pxx,'k')
plt.title('Pink Noise Frequency Domain')
plt.xlabel('Time (a.u.)')
plt.show()


# generate frequency-domain Gaussian with ifft
fwhm = 25 
hzGw =  np.linspace(0,srate,N) # vector for the fourier coeff
peak_freq = 50 # peak in frequency domain for the gaussian
hzGw_shift  = hzGw-peak_freq #shifted version of hzGw for generate gaussian in the a specific peak
gw = np.exp( -(4*np.log(2)*(hzGw_shift**2)) / fwhm**2 )

plt.plot(hzGw, gw,'k')
plt.title('Gaussian Frequency Domain')
plt.xlabel('Frequency')
plt.show()

ic = np.random.rand(N) * np.exp(1j*2*np.pi*np.random.rand(N)) # generate imaginary coeff, with random amplitudes
fc = ic * gw # add imaginary coeff to our gaussian

plt.plot(hzGw, fc,'k')
plt.title('Gaussian Frequency Domain')
plt.xlabel('Frequency')
plt.show()

g_sig = np.real( scipy.fftpack.ifft(fc) )*N # generate time domain signal with ifft

plt.plot(g_sig,'k')
plt.title('Gaussian Time Domain')
plt.xlabel('Time (a.u.)')
plt.show()

hzPxx = np.linspace(0, srate/2, int(np.floor(N/2)+1))
Pxx = np.abs(scipy.fftpack.fft(g_sig,N)/N)**2
Pxx = Pxx[0:len(hzPxx)] # taking only positive frequencies

plt.plot(hzPxx, Pxx,'k')
plt.title('Gaussian Frequency Domain')
plt.xlabel('Frequency')
plt.show()

# mix sig and noise
data = g_sig + noise

plt.plot(data)
plt.show()

hzPxx = np.linspace(0, srate/2, int(np.floor(N/2)+1))
Pxx = np.abs(scipy.fftpack.fft(data,N)/N)**2
Pxx = Pxx[0:len(hzPxx)] # taking only positive frequencies

plt.plot(hzPxx, Pxx,'k')
plt.xlabel('Frequency (norm.)')
plt.ylabel('Energy')
plt.title('Frequency-domain signal representation')
plt.show()











########################
######## XARRAY ########
########################


import xarray as xr



#### properties for the simulated data
n_trials = 10
n_channels = 5
n_times = 500

#### generate coordinates
conditions = ['Stimulus 0'] * 5 + ['Stimulus 1'] * 5
channels = [f"ch_{k}" for k in range(n_channels)]
times = np.arange(n_times) / 512.

#### create the (random) data
data_np = np.random.rand(n_trials, n_channels, n_times)

#### create the DataArray, dims are name of dimensions, coords is the labelling
data_xr = xr.DataArray(data_np, dims=['conditions', 'channels', 'times'], coords=[conditions, channels, times]) # dims have to be in the data order

#### xarray info
data_xr.name = 'GDR tuto' # ATTENTION if saved, the dataarray become a dataset and name a variable
data_xr.attrs = {"srate": 512., "subject": 0, "info": "subject was distracted at sample 1s"}

#### informations
data_xr.dims
data_xr.coords
data_xr.attrs

#### get values from attributes
data_xr.attrs['srate']

#### get values from dimensions, sauce Pandas
data_xr['conditions'].data # return an array
data_xr['channels'].data

#### indexing and slicing
data_xr.sel(times=slice(0., 0.5)) # to get everything that append during a specific time
data_xr.sel(channels='ch_1') # for coordinates that are not numerical

data_xr.sel(times=slice(0., 1.), channels=['ch_0', 'ch_1'], conditions='Stimulus 0') # for multi indexing

data_xr.isel(times=0, channels=3) # to select with integer as usual, isel for 'int selection'

#### mean
data_xr.mean(dim='conditions') # mean along the conditions dimension

#### groupby
groupby_object = data_xr.groupby('conditions') # here all the groupby values are stocked in the same object 
list_groupby = list(data_xr.groupby('conditions')) # better to access all dataarray

#### plot data directly
data_xr.mean(dim='conditions').sel(times=slice(0,1), channels='ch_0').plot()
plt.show()












########################################################
######## UPSAMPLE FOR REGULAR SPACED POINTS ########
########################################################

srate = 10 # originate srate
data  = np.array( [1, 4, 3, 6, 2, 19] )
npnts = len(data)
time  = np.arange(0,npnts)/srate

plt.plot(time,data,'ko-')
plt.show()

up_srate = 50 # new srate
up_npnts = int( len(data)*up_srate/srate )
up_time = np.arange(0,up_npnts)/up_srate
up_data = scipy.signal.resample(data,up_npnts)

plt.plot(up_time,up_data,'b^-',label='resample')
plt.plot(time,data,'ko',label='Original')
plt.legend()
plt.show()


    #### DOWNSAMPLE FOR REGULAR SPACED POINTS

srate = 100 # originate srate
time  = np.arange(-1,1,1/srate)
fwhm = .5
data  = np.exp( -(4*np.log(2)*(time**2))/fwhm**2)
npnts = len(data)

plt.plot(time,data,'ko-')
plt.show()

dw_srate = 10 # new srate
dw_npnts = int( len(data)*dw_srate/srate )
dw_time = np.arange(-1,1,1/dw_srate)
dw_data = scipy.signal.resample(data,dw_npnts)

plt.plot(dw_time,dw_data,'b^-',label='resample')
plt.plot(time,data,'ko',label='Original')
plt.legend()
plt.show()



    #### RESAMPLE FOR IREGULAR SPACED POINTS
srate = 10 # originate srate
x  = np.array( [1, 4, 3, 6, 2, 19] )
srate_resample = 10

f = scipy.interpolate.interp1d(x, y, kind='quadratic') # exist different type of kind
xnew = np.arange(x[0], x[-1], 1/srate_resample)
ynew = f(xnew)












################################
######## MEAN SMOOTHING ########
################################

filtsig = scipy.signal.savgol_filter(sig, srate*10, 3) # window size, polynomial order

# plot the noisy and filtered signals
plt.plot(time, sig, label='orig')
plt.plot(time, filtsig, label='filtered')

plt.legend()
plt.xlabel('Time (sec.)')
plt.ylabel('Amplitude')
plt.show()


















################################
######## DETREND ########
################################

# linear detrending
detsignal = scipy.signal.detrend(sig)















########################################
######## LINEAR REGRESSION ########
########################################


    #### simple regression

x = [29, 9, 10, 38, 16, 26, 50, 10, 30, 33, 43, 2, 39, 15, 44, 29, 41, 15, 24, 50]
y = [65, 7, 8, 76, 23, 56, 100, 3, 74, 48, 73, 0, 62, 37, 74, 40, 90, 42, 58, 100]

plt.scatter(x, y)
plt.show()

# polyfit
poly = np.polyfit(x,y,1) # compute linear model give back [a, b] if y = ax + b
model = poly[0]*range(0,max(x)) + poly[1]

plt.scatter(x, y)
plt.plot(model, c = 'r')
plt.show()

# linregress
lr = scipy.stats.linregress(x, y) # gives slope, intercept, rvalue, pvalue and std error
model = lr[0]*range(0,max(x)) + lr[1]

plt.scatter(x, y)
plt.plot(model, c = 'r')
plt.show()



a = np.array([[1, 2], [4, 5], [2, 7], [5, 7]])
b = np.array([[5], [14], [17], [20]])
x, residues, rank, s = np.linalg.lstsq(a, b)


    #### multiple regression

matdat = scipy.io.loadmat('EEG_RT_data.mat')
EEGdata = matdat['EEGdata']
frex = matdat['frex'][0]
rts = matdat['rts'][0]

b = np.zeros(len(frex)) # will gives us the intercept for every frequencies
for fi in range(0,len(frex)):
    
    model = np.polyfit(EEGdata[fi,:],rts,1)
    b[fi] = model[0]

plt.plot(frex,b)
plt.show()














########################
######## FFT ########
########################

    #### Classic FFT

# Parameters
x = sig
nfft = srate*10 # if zero padding nfft > len(x)
srate = 1000
hzPxx = np.linspace(0,srate/2,int(np.floor(nfft/2)))

Pxx = np.abs(scipy.fftpack.fft(x,nfft)/nfft)**2

plt.plot(hzPxx,Pxx[0:len(hzPxx)],'k')
plt.xlabel('Frequency (norm.)')
plt.ylabel('Energy')
plt.title('Frequency-domain signal representation')
plt.show()



    #### WELCH

# Parameters
x = sig
srate = 1000
nwind = int( 2*srate ) # window length in seconds*srate
nfft = nwind # if no zero padding nfft = nwind
noverlap = np.round(nwind/2) # number of points of overlap here 50%
hzPxx = np.linspace(0,srate/2,int(np.floor(nwind/2)+1)) # Hz vector, Hz resolution = nwind/2 because you take only points in positive domain. 
#The other half is for the negative domain. floor enable to get the low float not to have a decimal. And we add 1 to compensate for the
#floor but having all the points. int is here to have an integer
hannw = .5 - np.cos(2*np.pi*np.linspace(0,1,int(nwind)))/2 # hann window
Pxx = np.zeros(len(hzPxx)) # initialize the power matrix (windows x frequencies)
winonsets = np.arange(0,int(len(sig)-nwind),int(noverlap)) # window onset times

# Power computation
for wi in range(0,len(winonsets)):
    
    xw = x[ winonsets[wi]:winonsets[wi]+nwind ] # get a chunk of data from this time window
    xw = xw * hannw # apply Hann taper to data
    Xw = np.abs(scipy.fftpack.fft(xw,nfft)/nwind)**2 # compute its power
    Pxx = Pxx + Xw[0:len(hzPxx)] # we take only the positive part of Xw.

Pxx = Pxx / len(winonsets) # normalization

# plotting
plt.plot(hzPxx,Pxx/10,'r',label='Welch''s method')
plt.xlim([0,100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (a.u.)')
plt.legend()
plt.show()






    #### PYTHON S WELCH

# Parameters
srate = 1000
nwind = int( 2*srate ) # window length in seconds*srate
nfft = nwind # if no zero padding nfft = nwind
noverlap = np.round(nwind/2) # number of points of overlap here 50%
hannw = scipy.signal.windows.hann(nwind)

x = sig
hzPxx, Pxx = scipy.signal.welch(x,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)

plt.plot(hzPxx,Pxx)
#plt.semilogy(hzPxx,Pxx)
plt.xlim([0,100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (a.u.)')
plt.show()





















########################
######## FILTER ########
########################

    #### Band Pass Filter

# filter parameters
srate   = 1024 # hz
nyquist = srate/2
frange  = [20,45]
transw  = .1 #transition width = slope it is the % that we will take from the down frequency from which the filter will start
order   = int( 10*srate/frange[0] ) #length of the kernel, the higher the best is the frequency resolution and so the filter response. Too much would causes
#gain perturbation. 10 times the srate is ok. An other wway to think of it is to think of cycle at the filter frequency. The idea is also to have aproximatively
#3 cycles at the lower frequency in the frequency range.
# other idea that the order is the degree of polynome that are used to describe the filter

# order must be odd for firls function
if order%2==0:
    order += 1

# define filter shape
shape = [ 0, 0, 1, 1, 0, 0 ]
frex  = [ 0, frange[0]-frange[0]*transw, frange[0], frange[1], frange[1]+frange[1]*transw, nyquist ] # transform frange in a percent of nyquist

# filter kernel
fkern = scipy.signal.firls(order,frex,shape,fs=srate) # firls = finite impulse response from least squares

# compute the power spectrum of the filter kernel
filtpow = np.abs(scipy.fftpack.fft(fkern))**2
hz      = np.linspace(0,srate/2,int(np.floor(len(fkern)/2)+1))
filtpow = filtpow[0:len(hz)]

# Inspect the filter
plt.plot(fkern)
plt.xlabel('Time points')
plt.title('Filter kernel (firls)')
plt.show()

plt.plot(hz,filtpow,'ks-',label='Actual')
plt.plot(frex,shape,'ro-',label='Ideal')
plt.xlim([0,frange[0]*4])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter gain')
plt.legend()
plt.title('Frequency response of filter (firls)')
plt.show()



# implement the filter

# with manual filtering, reflection to avoid edge effects
#reflectdata = np.concatenate( (data[order:0:-1],data,data[-1:-1-order:-1]) ,axis=0)
# manual zero-phase-shift filter on the reflected signal
#reflectdata = signal.lfilter(fkern,1,reflectdata)
#reflectdata = signal.lfilter(fkern,1,reflectdata[::-1])
#reflectdata = reflectdata[::-1]
# now chop off the reflected parts
#fdata = reflectdata[order:-order]

# with filtfilt zero phase shift the function take care of everything
fdata1 = scipy.signal.filtfilt(fkern,1,data)


# inspect the result
plt.plot(range(0,N),data,'k',label='original')
plt.plot(range(0,N),fdata1,'b',label='filtered1')
plt.xlabel('Time (a.u.)')
plt.title('Time domain')
plt.legend()
plt.show()


plt.plot(hz,np.abs(scipy.fftpack.fft(data))**2,'k',label='Original')
plt.plot(hz,np.abs(scipy.fftpack.fft(fdata1))**2,'m',label='Filtered')
plt.legend()
plt.xlim([0,.5])
plt.xlabel('Frequency (norm.)')
plt.ylabel('Energy')
plt.title('Frequency domain')
plt.show()


    #### high pass filter

fcutoff = 5
transw  = .2
order   = np.round( 7*srate/fcutoff )

shape   = [ 1,1,0,0 ]
frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]

# filter kernel
filtkern = scipy.signal.firls(order,frex,shape,fs=srate)

# its power spectrum
filtkernX = np.abs(scipy.fftpack.fft(filtkern,n))**2

# inspect
plt.plot(np.arange(-order/2,order/2)/srate,filtkern,'k')
plt.xlabel('Time (s)')
plt.title('Filter kernel')
plt.show()

plt.plot(np.array(frex),shape,'r')
plt.plot(hzPxx,filtkernX[:len(hzPxx)],'k')
plt.xlim([0,60])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.title('Filter kernel spectrum')
plt.show()


# apply the filter
sigF = scipy.signal.filtfilt(filtkern,1,sig)

# power spectra of original and filtered signal
sigX = np.abs(scipy.fftpack.fft(sig)/n)**2
sigFX = np.abs(scipy.fftpack.fft(sigF)/n)**2

# inspect
plt.plot(time,sig,label='Signal')
plt.plot(time,sigF,label='Filtered')
plt.legend()
plt.xlabel('Time (sec.)')
plt.ylabel('Amplitude')
plt.show()

plt.plot(hzPxx,sigX[:len(hzPxx)],label='Signal')
plt.plot(hzPxx,sigFX[:len(hzPxx)],label='Filtered')
plt.xlim([0,srate/5])
plt.yscale('log')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.show()














################################
######## COMPLEX NUMBER ########
################################


# creating complex numbers
z = 4 + 3j # i is replaced by j be careful
z = complex(4,3)

# accessing different part of complex numbers
np.real(z)

np.imag(z)

# compute the magnitude
z_mag = np.abs(z)
z_mag = abs(z) # abs() directly works

# compute the angle
z_ang = np.angle(z)

# doing the conjugate
np.conj(z)

# computing the power of a complex number which is the same as power
# the 3 ways are exactly the same
z_Pxx1 = z*np.conj(z) 
z_Pxx2 = np.real(z)**2 + np.imag(z)**2
z_Pxx3 = np.abs(z)**2 # this is what we use the most to compute power

# plot complex number on a polar plan
plt.polar([0,z_ang],[0,z_mag],'r')
plt.show()


    #### DOT PRODUCT

srate = 1000
time = np.arange(-1,1,1/srate) 
ampl = 2
freq = 5
sw = ampl*np.sin(2*np.pi*freq*time) # generate a sinewave
gw = ampl*np.exp(-time**2/.1) # generate gaussian window
mw = sw * gw # generate morlets wavelet

plt.plot(mw)
plt.show()

np.dot(mw, sw) # the result is a real number and phase dependant


    #### COMPLEXE SINE WAVE

srate = 1000
time = np.arange(0,2-1/srate,1/srate)
freq = 5
ampl = 2
phase = np.pi/3

csw = ampl*np.exp(1j*(2*np.pi*freq*time+phase)) #complex sine wave

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(time, np.real(csw), np.imag(csw))
ax.legend()
plt.show()

    #### COMPLEX DOT PRODUCT

srate = 1000
time = np.arange(-1,1,1/srate) 
ampl = 2
freq = 5
csw = ampl*np.exp(1j*(2*np.pi*freq*time)) # generate a complex sinewave
gw = ampl*np.exp(-time**2/.1) # generate gaussian window
cmw = csw * gw # generate a complex morlets wavelet

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(time, np.real(cmw), np.imag(cmw))
ax.set_xlabel('Time')
ax.set_ylabel('Real Part')
ax.set_zlabel('Img Part')
ax.legend()
plt.show()

np.dot(cmw, csw) # the result is a complex number not phase dependent














################################
######## MORLET WAVELET ########
################################

srate = 1000
time = np.arange(-1,1+1/srate,1/srate) # this time vec for wavelet defines the frequency resolution
# This must have odd number of points to have the morlet wavelets centered
n = len(time) 
hzPxx = np.linspace(0,srate/2,int(np.floor(n/2)+1))


# parameters
freq = 4 # peak frequency
csw  = np.cos(2*np.pi*freq*time) # cosine wave used to have a centered wavelet
fwhm = .5 # full-width at half-maximum in seconds because timevec is in seconds
gw = np.exp( -(4*np.log(2)*time**2) / fwhm**2 ) # Gaussian

# Morlet wavelet
MorletWavelet = csw * gw # This is our Kernel

# amplitude spectrum
MorletWaveletPow = np.abs(scipy.fftpack.fft(MorletWavelet)/n)


# time-domain plotting
plt.subplot(211)
plt.plot(time,MorletWavelet,'k')
plt.xlabel('Time (sec.)')
plt.title('Morlet wavelet in time domain')

# frequency-domain plotting
plt.subplot(212)
plt.plot(hzPxx,MorletWaveletPow[0:len(hzPxx)],'k')
plt.xlim([0,freq*3])
plt.xlabel('Frequency (Hz)')
plt.title('Morlet wavelet in frequency domain')
plt.show()

    #### COMPLEX MORLET WAVELET

srate = 1000
time = np.arange(-1,1,1/srate) 
ampl = 2
freq = 5
csw = ampl*np.exp(1j*(2*np.pi*freq*time)) # generate a complex sinewave
fwhm = .5
gw = ampl*np.exp(-4*np.log(2)*time**2/ fwhm**2) 
cmw = csw * gw # generate a complex morlets wavelet

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(time, np.real(cmw), np.imag(cmw))
ax.set_xlabel('Time')
ax.set_ylabel('Real Part')
ax.set_zlabel('Img Part')
ax.legend()
plt.show()


    #### CONVOLUTION

sig = np.concatenate( (np.zeros(30),np.ones(2),np.zeros(20),np.ones(30),2*np.ones(10),np.zeros(30),-np.ones(10),np.zeros(40)) ,axis=0)
kernel  = np.exp( -np.linspace(-2,2,20)**2 )
kernel  = kernel/sum(kernel)
n = len(sig)

conv = np.convolve(sig,kernel,'same') # same enable to cutting of the wings

plt.subplot(311)
plt.plot(kernel,'k')
plt.xlim([0,n])
plt.title('Kernel')

plt.subplot(312)
plt.plot(sig,'k')
plt.xlim([0,n])
plt.title('Signal')

plt.subplot(313)
plt.plot( conv,'k')
plt.xlim([0,n])
plt.title('Convolution result')

plt.show()

    #### COMPLEX CONVOLUTION

# cmw
srate = 1000
time = np.arange(-1,1,1/srate) 
freq = 30
csw = np.exp(1j*(2*np.pi*freq*time)) # generate a complex sinewave

s = 7 / (2*np.pi*freq) # number of cycle that i want in my wavelet
gw = np.exp(-time**2/ (2*s**2)) 
cmw = csw * gw # generate a complex morlets wavelet

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(time, np.real(cmw), np.imag(cmw))
ax.set_xlabel('Time')
ax.set_ylabel('Real Part')
ax.set_zlabel('Img Part')
ax.legend()
plt.show()

fig = plt.figure()
plt.plot(time, np.real(cmw))
ax.set_xlabel('Time')
ax.set_ylabel('Real Part')
plt.show()

# signal
srate = 1000 # in Hz
n = srate*5 # 20 seconds
time  = np.arange(0,n)/srate # timevec
frex  = [ 10,20,45 ] # frequencies to include
sig = np.zeros(len(time)) # initiate signal vec

for fi in range(0,len(frex)):
    sig = sig + np.sin(2*np.pi*frex[fi]*time)

sig = sig + np.random.randn(len(sig)) # add some noise

plt.plot(time,sig)
plt.show()

# convolution
conv = np.convolve(sig,cmw,'same') # same enable to cutting of the wings, this result is complex

# features extraction

plt.subplot(311)
plt.plot(time, np.real(conv), 'k') # filtered signal
plt.title('Filtered signal')

plt.subplot(312)
plt.plot(time, abs(conv)**2 ,'k') # power
plt.title('Power')

plt.subplot(313)
plt.plot(time, np.angle(conv),'k') # phase
plt.title('Phase')

plt.show()












################################################
######## TIME FREQUENCY ANALYSIS ########
################################################




    #### TF analysis via spectrogram

# Parameters
x = sig
srate = 1000
nwind = int( 2*srate ) # window length in seconds*srate
nfft = nwind # if no zero padding nfft = nwind
noverlap = np.round(nwind/2) # number of points of overlap here 50%
hannw = scipy.signal.windows.hamming(nwind) # hann window

# TF computation
hzPxx,time,Pxx = scipy.signal.spectrogram(x, fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)

plt.pcolormesh(time,hzPxx,Pxx,vmin=0,vmax=1)
plt.ylim([0,100])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()





    #### TF analysis with Morlet Wavelets

# Create complex Morlet wavelets
# wavelet parameters
srate = 1000
wavetime = np.arange(-2,2,1/srate) 
nfrex = 50 # 50 frequencies
frex  = np.linspace(5,70,nfrex)
wavelets = np.zeros( (nfrex,len(wavetime)) ,dtype=complex) # matrix for all wavelets
ncycle = 7 # number of cycle in each wavelets

# create complex Morlet wavelet family
for fi in range(0,nfrex):
    # complex Morlet wavelet
    s = ncycle / (2*np.pi*frex[fi]) # number of cycle that i want in my wavelet
    gw = np.exp(-wavetime**2/ (2*s**2)) 
    sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
    mw =  gw * sw
    # fill the matrix
    wavelets[fi,:] = mw
    
# plot all the wavelets
plt.pcolormesh(wavetime,frex,np.real(wavelets))
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.title('Real part of wavelets')
plt.show()

# load data
braindat = scipy.io.loadmat('data4TF')
timevec = braindat['timevec'][0]
srate = braindat['srate'][0]
data = braindat['data'][0]

plt.plot(data)
plt.show()


    #### run manual convolution through convolution theorem

# convolution parameters
nconv = len(timevec) + len(wavetime) - 1 # M+N-1
halfk = int( np.floor(len(wavetime)/2) )

# Fourier spectrum of the signal
dataX = scipy.fftpack.fft(data,nconv)

# initialize time-frequency matrix
tf = np.zeros( (nfrex,len(timevec)) )

# convolution per frequency
for fi in range(0,nfrex):
    
    # FFT of the wavelet
    waveX = scipy.fftpack.fft(wavelets[fi,:],nconv)
    # amplitude-normalize the wavelet
    waveX = waveX/np.max(waveX) # This to prevent power injection of low frequencies Morlet Wavelets. Thus all wavelets have the same power.
    
    # convolution using the convolution theorem
    convres = scipy.fftpack.ifft( waveX*dataX )
    # trim the "wings"
    convres = convres[halfk-1:-halfk]
    
    # extract power from complex signal
    tf[fi,:] = np.abs(convres)**2

plt.pcolormesh(timevec,frex,tf,vmin=0,vmax=1e3)
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.title('Time-frequency power')
plt.show()


    #### run python convolution

# initialize time-frequency matrix
tf = np.zeros( (nfrex,len(data)) )

# convolution per frequency
for fi in range(0,nfrex):
    
    tf[fi,:] = abs(scipy.signal.fftconvolve(data, wavelets[fi], 'same'))**2 # here we use the convolution theorem by fft computation, the function trim the wing by itself

plt.pcolormesh(timevec,frex,tf,vmin=0,vmax=1e8)
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.title('Time-frequency power')
plt.show()




















########################################
######## POWER NORMALIZATION ########
########################################

    #### Generate signal 1/f

# generate 1/f noise
srate  = 1000
time = np.arange(-1,2, 1/srate)
N   = len(time)
width = 200
pink_n = np.random.rand(N) * np.exp(-np.arange(0,N)/width) # generate 1/f pink noise with inverse (negative) exponential
fc  = pink_n * np.exp(2*np.pi*1j*np.random.rand(len(pink_n))) # generate fourier coeff based on noise, we add imaginary part 
amp = 8000 # amplitude for noise
noise = amp * np.real(scipy.fftpack.ifft(fc)) # take the ifft to go in time domain 

# generate frequency-domain Gaussian with ifft
fwhm = 25 
hzGw =  np.linspace(0,srate,N) # vector for the fourier coeff
peak_freq = 50 # peak in frequency domain for the gaussian
hzGw_shift  = hzGw-peak_freq #shifted version of hzGw for generate gaussian in the a specific peak
gw = np.exp( -(4*np.log(2)*(hzGw_shift**2)) / fwhm**2 )

ic = np.random.rand(N) * np.exp(1j*2*np.pi*np.random.rand(N)) # generate imaginary coeff, with random amplitudes
fc = ic * gw # add imaginary coeff to our gaussian

g_sig = np.real( scipy.fftpack.ifft(fc) )*N # generate time domain signal with ifft

# mix sig and noise
data = g_sig + noise

plt.plot(time,data)
plt.show()

    #### Add non stationnary feature

# wavelet
srate = 1000
wavetime = time
fwhm = .5
freq = 50 # 50 frequencies
gw = 2 * np.exp(-(4*np.log(2) * wavetime**2/ (fwhm**2)))
sw = np.sin(2*np.pi*freq*wavetime)
mw =  gw * sw

data = data + mw*20

plt.plot(time,data)
plt.show()


    #### Morlet wavelets

# wavelet parameters
srate = 1000
wavetime = np.arange(-2,2,1/srate) 
nfrex = 50 # 50 frequencies
frex  = np.linspace(5,70,nfrex)
wavelets = np.zeros( (nfrex,len(wavetime)) ,dtype=complex) # matrix for all wavelets
ncycle = 7 # number of cycle in each wavelets

# create complex Morlet wavelet family
for fi in range(0,nfrex):
    # complex Morlet wavelet
    s = ncycle / (2*np.pi*frex[fi]) # number of cycle that i want in my wavelet
    gw = np.exp(-wavetime**2/ (2*s**2)) 
    sw = np.exp(1j*2*np.pi*frex[fi]*wavetime)
    mw =  gw * sw
    # fill the matrix
    wavelets[fi,:] = mw
    
    #### TF

tf = np.zeros( (nfrex,len(data)) )
for fi in range(0,nfrex):
    
    tf[fi,:] = abs(scipy.signal.fftconvolve(data, wavelets[fi], 'same'))**2 # here we use the convolution theorem by fft computation, the function trim the wing by itself

plt.pcolormesh(time,frex,tf)
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.title('Time-frequency power')
plt.show()

    #### Baseline computation

baseline_i = [0, 500] # number of points
baselineX = []

# average power for baseline
for fi in range(0,nfrex):

    baselineX.append(np.ndarraymean(tf[fi,baseline_i[0]:baseline_i[1]]))

plt.plot(frex,baselineX)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Time-frequency power')
plt.show()

    #### Baseline implementation

db_tf = np.zeros((nfrex,len(time)))
for fi in range(0,nfrex):

    activity = tf[fi,:]
    baseline = baselineX[fi]

    db_tf[fi,:] = 10*np.log10(activity/baseline)

plt.pcolormesh(time,frex,db_tf)
#plt.pcolormesh(time,frex,db_tf,vmax = db_tf.max()*3 )
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.title('Time-frequency power')
plt.show()















########################################
######## HILBERT TRANSFORM ########
########################################

    #### Band Pass Filter

# filter parameters
srate   = 1000 # hz
nyquist = srate/2
frange  = [20,25]
transw  = .1 #transition width = slope
order   = int( 10*srate/frange[0] ) #length of the kernel, the higher the best is the frequency resolution and so the filter response. Too much would causes
#gain perturbation. 10 times the srate is ok. An other wway to think of it is to think of cycle at the filter frequency. The idea is also to have aproximatively
#3 cycles at the lower frequency in the frequency range.

# order must be odd for firls function
if order%2==0:
    order += 1

# define filter shape
shape = [ 0, 0, 1, 1, 0, 0 ]
frex  = [ 0, frange[0]-frange[0]*transw, frange[0], frange[1], frange[1]+frange[1]*transw, nyquist ] # transform frange in a percent of nyquist

# filter kernel
fkern = scipy.signal.firls(order,frex,shape,fs=srate) # firls = finite impulse response from least squares

# compute the power spectrum of the filter kernel
filtpow = np.abs(scipy.fftpack.fft(fkern))**2
hz      = np.linspace(0,srate/2,int(np.floor(len(fkern)/2)+1))
filtpow = filtpow[0:len(hz)]

# Inspect the filter
plt.plot(fkern)
plt.xlabel('Time points')
plt.title('Filter kernel (firls)')
plt.show()

plt.plot(hz,filtpow,'ks-',label='Actual')
plt.plot(frex,shape,'ro-',label='Ideal')
plt.xlim([0,frange[0]*4])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter gain')
plt.legend()
plt.title('Frequency response of filter (firls)')
plt.show()

# generate random signal

sig = np.random.randn(srate*4)

# implement the filter

f_sig = scipy.signal.filtfilt(fkern,1,sig)


# inspect the result
plt.plot(sig,'k',label='original')
plt.plot(f_sig,'m',label='filtered')
plt.xlabel('Time (a.u.)')
plt.title('Time domain')
plt.legend()
plt.show()

hz = np.linspace(0,srate/2,int(np.floor(len(sig)/2)))
hzPxx = np.abs(scipy.fftpack.fft(sig))**2
plt.plot(hz,hzPxx[0:int(len(sig)/2)],'k',label='Original')

hzPxx = np.abs(scipy.fftpack.fft(f_sig))**2
plt.plot(hz,hzPxx[0:int(len(f_sig)/2)],'m',label='Filtered')

plt.legend()
plt.xlabel('Frequency (norm.)')
plt.ylabel('Energy')
plt.title('Frequency domain')
plt.show()

# implement the Hilbert transform

ht = scipy.signal.hilbert(f_sig,len(f_sig))

# inspect results
plt.subplot(311)
plt.plot(np.real(ht),'k')
plt.title('Real Part')

plt.subplot(312)
plt.plot(abs(ht),'k')
plt.xlabel('Frequency (Hz)')
plt.title('Magnitude')

plt.subplot(313)
plt.plot(np.angle(ht),'k')
plt.title('Phase')
plt.show()



















################################################
######## INTER TRIAL PHASE CLUSTERING ########
################################################

    #### Phase clustering for one time point 
# Angle signal generation
circ_prop = .5 # proportion of the circle to fill
n = 100 # number of trials

simdata = np.random.rand(n) * 2*np.pi * circ_prop # generate phase angle distribution for one time point in trial

# Compute ITCP
itpc_amp = abs(np.mean(np.exp(1j*simdata))) # we compute the mean of the amplitude
itpc_phase = np.angle(np.mean(np.exp(1j*simdata))) # and the phase vector

plt.polar([np.zeros(n),simdata],[np.zeros(n),np.ones(n)],'b')
plt.polar([0,itpc_phase],[0,itpc_amp],'r',linewidth=4)
plt.show()

count, bins = np.histogram(simdata)
fig, ax = plt.subplots()
ax.hist(simdata, 20)
plt.xlim(0, 2*np.pi)
plt.show()


    #### Phase clustering for real trials 
# Extract data from .mt file
matlab = scipy.io.loadmat('sampleEEGdata.mat') # open as a dict, use .keys to select key to open 
matlabEEG = matlab['EEG'] # use .dtype to select what to open
srate_mat = matlabEEG['srate'][0][0][0][0]
timevec = matlabEEG['times'][0][0][0]

data = matlabEEG['data'][0][0][0] # this is for one channel 99 trials
data = data.transpose() # to have a matrix trials by time

plt.plot(data[:,0])
plt.show()

# Generate Morlet Wavelet
srate = srate_mat
wavetime = np.arange(-2,2,1/srate) 
nfrex = 40
frex  = np.linspace(2,30,nfrex)
wavelets = np.zeros( (nfrex,len(wavetime)) ,dtype=complex)
ncycle = 7

for fi in range(0,nfrex):
    s = ncycle / (2*np.pi*frex[fi])
    gw = np.exp(-wavetime**2/ (2*s**2)) 
    sw = np.exp(1j*2*np.pi*frex[fi]*wavetime)
    mw =  gw * sw
    wavelets[fi,:] = mw

# Convolve
convmat = np.zeros( (data.shape[0],data.shape[1],nfrex) ,dtype=complex) # convolution matrice

for ti in range(0,data.shape[0]):

    for fi in range(0,nfrex):
    
        convmat[ti,:,fi] = scipy.signal.fftconvolve(data[ti,:], wavelets[fi], 'same') # here we use the convolution theorem by fft computation, the function trim the wing by itself

# Average phase
angle_convmat = np.angle(convmat) # keep only angle information
complex_angle = np.exp(1j*angle_convmat) # transform angle in vector to fit in the polar plane
itpc = np.abs(np.mean(angle_convmat,0)) # perform the magnitude of the average angle that represent the "intensity" of the coupling
itpc = itpc.transpose() # to have a matrix frex by time

# ITCP pcolormesh
plt.pcolormesh(timevec,frex,itpc, vmin=itpc.max()/10, vmax=itpc.max()/1.5)
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.title('ITPC')
plt.show()


















################################################
######## INTER SITE PHASE CLUSTERING ########
################################################


# open matlab data, these data are channel/sig/trial

data = scipy.io.loadmat('v1_laminar.mat')['csd']
srate = scipy.io.loadmat('v1_laminar.mat')['srate'][0]
time = scipy.io.loadmat('v1_laminar.mat')['timevec'][0]

# channel for connectivity, we will work on the first trial
chan1 = 0
chan2 = 7

trial_num = 0

# complex convolution
freq = 8
time_cmw = np.arange(-1.5,1.5+1/srate,1/srate)
ncycle = 8
s = ncycle/(2*np.pi*freq)
cmw = np.exp(1j*2*np.pi*freq*time_cmw) * np.exp(-time_cmw**2./(2*s**2))

# choose signal
x = data[chan1,:,trial_num]
y = data[chan2,:,trial_num]

# convolve to get the analytic signal
as1 = scipy.signal.fftconvolve(x,cmw,'same') 
as2 = scipy.signal.fftconvolve(y,cmw,'same') 


# inspect results
plt.subplot(311)
plt.plot(time,np.real(as1),'k')
plt.plot(time,np.real(as2),'r')
plt.title('Real Part')

plt.subplot(312)
plt.plot(abs(as1),'k')
plt.plot(abs(as2),'r')
plt.xlabel('Frequency (Hz)')
plt.title('Magnitude')

plt.subplot(313)
plt.plot(time,np.angle(as1),'k')
plt.plot(time,np.angle(as2),'r')
plt.title('Phase')
plt.show()


# compute synchro
diff_phase = np.angle(as2) - np.angle(as1) # phase temporal serie from the 2 signals
euler_phase_diff = np.exp(1j*diff_phase) # to position phase values to the complex plan
mean_complex_vector = np.mean(euler_phase_diff) # mean for all phase value
phase_synchro = np.abs(mean_complex_vector) # phase synchro value

plt.polar([np.zeros((np.size(diff_phase)), dtype='complex'),diff_phase],[np.zeros(np.size(diff_phase)),np.ones(np.size(diff_phase))],'k')
plt.polar([np.zeros(1),np.angle(mean_complex_vector)],[np.zeros(1),np.abs(mean_complex_vector)],'r', linewidth=4)
plt.show()























########################################
######## CURRENT SOURCE DENSITY ########
########################################



# load data
data = scipy.io.loadmat('sampleEEGdata.mat')['EEG']['data'][0][0]
data = np.mean(data,2)
srate = scipy.io.loadmat('sampleEEGdata.mat')['EEG']['srate'][0][0][0][0]

time = scipy.io.loadmat('sampleEEGdata.mat')['EEG']['times'][0][0][0]

chan_list_mat = scipy.io.loadmat('sampleEEGdata.mat')['EEG']['chanlocs'][0][0]['labels'][0]
chan_list = [chan_list_mat[nchan][0] for nchan in range(np.size(chan_list_mat))]

eeg_dict = dict()
for nchan in range(np.size(chan_list)):
    print(nchan/len(chan_list))
    x = scipy.io.loadmat('sampleEEGdata.mat')['EEG']['chanlocs'][0][0]['X'][0][nchan][0][0]
    y = scipy.io.loadmat('sampleEEGdata.mat')['EEG']['chanlocs'][0][0]['Y'][0][nchan][0][0]
    z = scipy.io.loadmat('sampleEEGdata.mat')['EEG']['chanlocs'][0][0]['Z'][0][nchan][0][0]
    eeg_dict[chan_list[nchan]] = [y, x, z]

montage = mne.channels.make_dig_montage(ch_pos=eeg_dict)
info =  mne.create_info(chan_list, srate, ch_types='eeg', verbose=None, montage=montage)
raw = mne.io.RawArray(data,info)


# laplacian computation
from surface_laplacian import surface_laplacian
raw_lap_sl = surface_laplacian(raw=raw, m=4, leg_order=50, smoothing=1e-5) # MXC way

raw_lap_mne = raw.copy()
raw_lap_mne = mne.preprocessing.compute_current_source_density(raw_lap_mne, stiffness=2, lambda2=0, n_legendre_terms=50) # MNE way

# verif
time2plot = np.where(time == 250)[0][0] # for 250ms

fig, axs = plt.subplots(ncols=3)
ax = axs[0]
mne.viz.plot_topomap(raw.get_data()[:,time2plot], raw.info, axes=axs[0], names=chan_list, show_names=True, show=False)
ax.set_title('raw')
ax = axs[1]
mne.viz.plot_topomap(raw_lap_sl.get_data()[:,time2plot], raw.info, axes=axs[1], names=chan_list, show_names=True, show=False)
ax.set_title('raw_lap_sl')
ax = axs[2]
mne.viz.plot_topomap(raw_lap_mne.get_data()[:,time2plot], raw.info, axes=axs[2], names=chan_list, show_names=True, show=False)
ax.set_title('raw_lap_mne')
plt.show()


# chan of interest
ch2plot = 'Cz'
ch_i = chan_list.index(ch2plot)

# signal
voltERP = data[ch_i,:]
lapERP = raw_lap_sl.get_data()[ch_i,:]
lapERP_mne = raw_lap_mne.get_data()[ch_i,:]

# verify
plt.plot(time,(voltERP-np.mean(voltERP))/np.std(voltERP), label='voltERP')
plt.plot(time,(lapERP-np.mean(lapERP))/np.std(lapERP), label='lapERP')
plt.plot(time,(lapERP_mne-np.mean(lapERP_mne))/np.std(lapERP_mne), label='lapERP_mne')
plt.xlim(-300,1200)
plt.legend()
plt.show()













########################################
######## PHASE LAG INDEX ########
########################################



# generate phase population from the difference between two electrodes 
N = 200
pfilled = .3
centphase = np.pi/4

phase_angles = np.linspace(centphase-pfilled*np.pi,centphase+pfilled*np.pi,N)


# compute PLI:
cdd = np.exp(1j*phase_angles) # "eulerize" the phase angle differences
cdi = np.imag(cdd) # project the phase angle differences onto the imaginary axis
cdis = np.sign(cdi) # take the sign of those projections
cdism = np.mean(cdis) # take the average sign
pli = np.abs(cdism) # we can about the magnitude of the average


# for reference, compute ISPC
ispc = np.abs(np.mean(cdd))

# verify
plt.polar([np.zeros(N),phase_angles],[np.zeros(N),np.ones(N)],'b')
plt.title('PLI = '+str(pli)+', ITPC = '+str(ispc))
plt.show()


















########################################################
######## PHASE LAG VERSUS PHASE CLUSTERING ########
########################################################



# signal parameters
srate = 1000
time = np.arange(0,5*srate-1)/srate

frex = 7.3
phaselag = .05 * (2*np.pi)
noiselevel = 10

sig1 = noiselevel * np.sin(2*np.pi*frex*time) + np.random.randn(np.size(time))
sig2 = noiselevel * np.sin(2*np.pi*frex*time + phaselag) + np.random.randn(np.size(time))


# verify
plt.plot(time,sig1, label='sig1')
plt.plot(time,sig2, label='sig2')
plt.xlabel('Time (s)')
plt.legend
plt.show()

# compute ISPC and PLI

# frequency parameters
min_freq =  2
max_freq = 15
num_frex = 50

srate = 1000
wavetime = np.arange(-2,2,1/srate) 
frex  = np.linspace(min_freq,max_freq,num_frex)
wavelets = np.zeros((num_frex,len(wavetime)) ,dtype=complex)
ncycle = 7 

for fi in range(0,num_frex):
    s = ncycle / (2*np.pi*frex[fi])
    gw = np.exp(-wavetime**2/ (2*s**2)) 
    sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
    mw =  gw * sw
    wavelets[fi,:] = mw

# initialize output time-frequency data
ispc = np.zeros((num_frex))
pli  = np.zeros((num_frex))

# convolution per frequency
for fi in range(0,num_frex):
    
    as1 = scipy.signal.fftconvolve(sig1, wavelets[fi], 'same')
    as2 = scipy.signal.fftconvolve(sig2, wavelets[fi], 'same')

    # collect "eulerized" phase angle differences
    cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
    
    # compute ISPC and PLI (and average over trials!)
    ispc[fi] = np.abs(np.mean(cdd))
    pli[fi] = np.abs(np.mean(np.sign(np.imag(cdd))))


plt.plot(frex,ispc, label='ISPC')
plt.plot(frex,pli, label='PLI')
plt.ylim(0,1.1)
plt.xlabel('Frequency (Hz)'), plt.ylabel('Synchronization strength')
plt.legend()
plt.show()

























########################################
######## GRANGER CAUSALITY ########
########################################


from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR



maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

grangers_causation_matrix(df, variables = df.columns)        


data_GC = np.stack((data[chan1i,:], data[chan2i,:]), axis=1)
test = 'ssr_chi2test'
test_result = grangercausalitytests(data_GC, maxlag=order, verbose=False)
p_values = [round(test_result[i+1][0][test][1],4) for i in range(order)]

data_GC = np.stack((data[chan2i,:], data[chan1i,:]), axis=1)
test = 'ssr_chi2test'
test_result = grangercausalitytests(data_GC, maxlag=order, verbose=False)
p_values = [round(test_result[i+1][0][test][1],4) for i in range(order)]


test_result[1][0][test][1]








#### Univariate AR

x = [1]
for i in range(99):
    x.append(1.1*x[i]) # here model with only one order

x = [1, 2]
for i in range(98):
    x.append(1.1*x[i+1] - .9*x[i]) # here model with only one order



plt.plot(x)
plt.show()

order = 2 # same thing as lags, number of points used to compute the regression, i. e. the precedent points that weight the next point
res = AutoReg(x, lags = order).fit()

res.summary()
res.fittedvalues # fitted values minus the number of lags
res.resid # residuals
#### for the next result the first number is the intercept
res.pvalues # t-test for coeff
res.params # coeff
res.bse # standard errors of the estimated parameters
res.bic # Bayesian information criterion (BIC) for model selectionthe model with the lowest BIC is preferred.

bic_list = []
bse_list = []
for order in [1, 2, 3, 4, 5]:
    res = AutoReg(x, lags = order).fit()
    bic_list.append(res.bic) 
    bse_list.append(res.bse)

plt.plot(bse_list)
plt.show()

plt.plot(x)
plt.plot(res.fittedvalues)
plt.plot(res.resid)
plt.plot(model)
plt.show()

#### Bivariate AR


srate = 1000
time  = np.arange(0, 1, 1/srate)
x = 0.39*np.sin(2*np.pi*10*time) + .7*np.sin(2*np.pi*2*time) + np.random.randn(len(time))/10
y = np.array([0, 0])
for i in range(len(x)-2):
    y = np.append(y, -.8*x[i] + 1.2*x[i+1])

plt.plot(x)
plt.plot(y)
plt.show()



data_VAR = np.stack((x, y), axis=1)

order = 2
res = VAR(data_VAR).fit(order)
res.summary()
res.fittedvalues
res.pvalues # t-test for coeff
res.resid # residuals
res.bse # std for residuals, first is intercept
res.params # coeff, first is intercept








# load data
data = scipy.io.loadmat('sampleEEGdata.mat')['EEG']['data'][0][0]
data = np.mean(data,2)
srate = scipy.io.loadmat('sampleEEGdata.mat')['EEG']['srate'][0][0][0][0]

time = scipy.io.loadmat('sampleEEGdata.mat')['EEG']['times'][0][0][0]

chan_list_mat = scipy.io.loadmat('sampleEEGdata.mat')['EEG']['chanlocs'][0][0]['labels'][0]
chan_list = [chan_list_mat[nchan][0] for nchan in range(np.size(chan_list_mat))]


# define channels to compute granger synchrony between
chan1name = 'FCz'
chan2name = 'O1'

chan1i = chan_list.index(chan1name)
chan2i = chan_list.index(chan2name)

# define autoregression parameters (can leave as default for now)
order = 14

res = AutoReg(data[chan1i,:], lags = order).fit()
res.summary()
res.fittedvalues
res.pvalues # t-test for coeff
res.resid # residuals
res.bse # std for residuals, first is intercept
res.params # coeff, first is intercept

plt.plot(time, data[chan1i,:], label='orig')
plt.plot(time, np.insert(res.fittedvalues, 0, np.zeros((order))), label='AR model')
plt.legend()
plt.show()

data_GC = np.stack((data[chan1i,:], data[chan2i,:]), axis=1)

res = grangercausalitytests(data_GC, maxlag=1, verbose=False)






