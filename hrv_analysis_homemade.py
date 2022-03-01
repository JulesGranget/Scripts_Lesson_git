



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal 
import os
import pandas as pd
from bycycle.cyclepoints import find_extrema

debug = False


########################################
######## HRV ANALYSIS HOMEMADE ########
########################################

#### params
def get_params_hrv_homemade(srate_resample_hrv):
    
    nwind_hrv = int( 128*srate_resample_hrv )
    nfft_hrv = nwind_hrv
    noverlap_hrv = np.round(nwind_hrv/90)
    win_hrv = scipy.signal.windows.hann(nwind_hrv)
    f_RRI = (.1, .5)

    return nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv, f_RRI


#### RRI, IFR
#ecg_i, ecg_cR, srate, srate_resample = ecg_i, ecg_cR, srates['ecg'], srate_resample_hrv
def get_RRI_IFR(ecg_i, ecg_cR, srate, srate_resample) :

    cR_sec = ecg_cR # cR in sec

    # RRI computation
    RRI = np.diff(cR_sec)
    RRI = np.insert(RRI, 0, np.median(RRI))
    IFR = (1/RRI)

    # interpolate
    f = scipy.interpolate.interp1d(cR_sec, RRI, kind='quadratic')
    cR_sec_resample = np.arange(cR_sec[0], cR_sec[-1], 1/srate_resample)
    RRI_resample = f(cR_sec_resample)

    #plt.plot(cR_sec, RRI, label='old')
    #plt.plot(cR_sec_resample, RRI_resample, label='new')
    #plt.legend()
    #plt.show()

    return RRI, RRI_resample, IFR

def get_fig_RRI_IFR(ecg_i, ecg_cR, RRI, IFR, srate, srate_resample):

    cR_sec = ecg_cR # cR in sec
    times = np.arange(0,len(ecg_i))/srate # in sec

    f = scipy.interpolate.interp1d(cR_sec, RRI, kind='quadratic')
    cR_sec_resample = np.arange(cR_sec[0], cR_sec[-1], 1/srate_resample)
    RRI_resample = f(cR_sec_resample)

    fig, ax = plt.subplots()
    ax = plt.subplot(411)
    plt.plot(times, ecg_i)
    plt.title('ECG')
    plt.ylabel('a.u.')
    plt.xlabel('s')
    plt.vlines(cR_sec, ymin=min(ecg_i), ymax=max(ecg_i), colors='k')
    plt.subplot(412, sharex=ax)
    plt.plot(cR_sec, RRI)
    plt.title('RRI')
    plt.ylabel('s')
    plt.subplot(413, sharex=ax)
    plt.plot(cR_sec_resample, RRI_resample)
    plt.title('RRI_resampled')
    plt.ylabel('Hz')
    plt.subplot(414, sharex=ax)
    plt.plot(cR_sec, IFR)
    plt.title('IFR')
    plt.ylabel('Hz')
    #plt.show()

    # in this plot one RRI point correspond to the difference value between the precedent RR
    # the first point of RRI is the median for plotting consideration

    return fig

    

#### LF / HF

#RRI_resample, srate_resample, nwind, nfft, noverlap, win = RRI_resample, srate_resample, nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv
def get_PSD_LF_HF(RRI_resample, srate_resample, nwind, nfft, noverlap, win, VLF, LF, HF):

    # DETREND
    RRI_detrend = RRI_resample-np.median(RRI_resample)

    # FFT WELCH
    hzPxx, Pxx = scipy.signal.welch(RRI_detrend, fs=srate_resample, window=win, nperseg=nwind, noverlap=noverlap, nfft=nfft)

    AUC_LF = np.trapz(Pxx[(hzPxx>VLF) & (hzPxx<LF)])
    AUC_HF = np.trapz(Pxx[(hzPxx>LF) & (hzPxx<HF)])
    LF_HF_ratio = AUC_LF/AUC_HF

    return AUC_LF, AUC_HF, LF_HF_ratio, hzPxx, Pxx


def get_fig_PSD_LF_HF(Pxx, hzPxx, VLF, LF, HF):

    # PLOT
    fig = plt.figure()
    plt.plot(hzPxx,Pxx)
    plt.ylim(0, np.max(Pxx[hzPxx>0.01]))
    plt.xlim([0,.6])
    plt.vlines([VLF, LF, HF], ymin=min(Pxx), ymax=max(Pxx), colors='r')
    #plt.show()
    
    return fig


#### SDNN, RMSSD, NN50, pNN50
# RR_val = RRI
def get_stats_descriptors(RR_val) :
    SDNN = np.std(RR_val)

    RMSSD = np.sqrt(np.mean((np.diff(RR_val)*1e3)**2))

    NN50 = []
    for RR in range(len(RR_val)) :
        if RR == len(RR_val)-1 :
            continue
        else :
            NN = abs(RR_val[RR+1] - RR_val[RR])
            NN50.append(NN)

    NN50 = np.array(NN50)*1e3
    pNN50 = np.sum(NN50>50)/len(NN50)

    return SDNN, RMSSD, NN50, pNN50

#SDNN_CV, RMSSD_CV, NN50_CV, pNN50_CV = get_stats_descriptors(RRI_CV)


#### Poincarré

def get_poincarre(RRI):
    RRI_1 = RRI[1:]
    RRI_1 = np.append(RRI_1, RRI[-1]) 

    SD1_val = []
    SD2_val = []
    for RR in range(len(RRI)) :
        if RR == len(RRI)-1 :
            continue
        else :
            SD1_val_tmp = (RRI[RR+1] - RRI[RR])/np.sqrt(2)
            SD2_val_tmp = (RRI[RR+1] + RRI[RR])/np.sqrt(2)
            SD1_val.append(SD1_val_tmp)
            SD2_val.append(SD2_val_tmp)

    SD1 = np.std(SD1_val)
    SD2 = np.std(SD2_val)
    Tot_HRV = SD1*SD2*np.pi

    return SD1, SD2, Tot_HRV


        
def get_fig_poincarre(RRI):

    RRI_1 = RRI[1:]
    RRI_1 = np.append(RRI_1, RRI[-1]) 

    fig = plt.figure()
    plt.scatter(RRI, RRI_1)
    plt.xlabel('RR (ms)')
    plt.ylabel('RR+1 (ms)')
    plt.title('Poincarré ')
    plt.xlim(.600,1.)
    plt.ylim(.600,1.)

    return fig
    
#### DeltaHR

#RRI, srate_resample, f_RRI, condition = result_struct[keys_result[0]][1], srate_resample, f_RRI, cond 
def get_dHR(RRI_resample, srate_resample, f_RRI):
    
    times = np.arange(0,len(RRI_resample))/srate_resample

        # stairs method
    #RRI_stairs = np.array([])
    #len_cR = len(cR) 
    #for RR in range(len(cR)) :
    #    if RR == 0 :
    #        RRI_i = cR[RR+1]/srate - cR[RR]/srate
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(cR[RR+1]))])
    #    elif RR != 0 and RR != len_cR-1 :
    #        RRI_i = cR[RR+1]/srate - cR[RR]/srate
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(cR[RR+1] - cR[RR]))])
    #    elif RR == len_cR-1 :
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(len(ecg) - cR[RR]))])


    peaks, troughs = find_extrema(RRI_resample, srate_resample, f_RRI)
    peaks_RRI, troughs_RRI = RRI_resample[peaks], RRI_resample[troughs]
    peaks_troughs = np.stack((peaks_RRI, troughs_RRI), axis=1)

    fig_verif = plt.figure()
    plt.plot(times, RRI_resample)
    plt.vlines(peaks/srate_resample, ymin=min(RRI_resample), ymax=max(RRI_resample), colors='b')
    plt.vlines(troughs/srate_resample, ymin=min(RRI_resample), ymax=max(RRI_resample), colors='r')
    #plt.show()

    dHR = np.diff(peaks_troughs/srate_resample, axis=1)*1e3

    fig_dHR = plt.figure()
    ax = plt.subplot(211)
    plt.plot(times, RRI_resample*1e3)
    plt.title('RRI')
    plt.ylabel('ms')
    plt.subplot(212, sharex=ax)
    plt.plot(troughs/srate_resample, dHR)
    plt.hlines(np.median(dHR), xmin=min(times), xmax=max(times), colors='m', label='median = {:.3f}'.format(np.median(dHR)))
    plt.legend()
    plt.title('dHR')
    plt.ylabel('ms')
    plt.vlines(peaks/srate_resample, ymin=0, ymax=0.01, colors='b')
    plt.vlines(troughs/srate_resample, ymin=0, ymax=0.01, colors='r')
    plt.tight_layout()
    #plt.show()


    return fig_verif, fig_dHR


def ecg_analysis_homemade(ecg_i, srate, srate_resample_hrv, fig_token=False):

    #### load params
    nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv, f_RRI = get_params_hrv_homemade(srate_resample_hrv)

    #### load cR
    ecg_cR = scipy.signal.find_peaks(ecg_i, distance=srate*0.5)[0]
    ecg_cR = ecg_cR/srate

    #### verif
    if debug:
        times = np.arange(ecg_i.shape[0])/srate
        plt.plot(times, ecg_i)
        plt.vlines(ecg_cR, ymin=np.min(ecg_i) ,ymax=np.max(ecg_i), colors='r')
        plt.show()


    #### initiate metrics names
    res_list = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2', 'HRV_S']

    #### RRI
    RRI, RRI_resample, IFR = get_RRI_IFR(ecg_i, ecg_cR, srate, srate_resample_hrv)

    HRV_MeanNN = np.mean(RRI)
    
    #### PSD
    VLF, LF, HF = .04, .15, .4
    AUC_LF, AUC_HF, LF_HF_ratio, hzPxx, Pxx = get_PSD_LF_HF(RRI_resample, srate_resample_hrv, nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv, VLF, LF, HF)

    #### descriptors
    SDNN, RMSSD, NN50, pNN50 = get_stats_descriptors(RRI)

    #### poincarré
    SD1, SD2, Tot_HRV = get_poincarre(RRI)

    #### df
    res_tmp = [HRV_MeanNN*1e3, SDNN*1e3, RMSSD, pNN50*100, AUC_LF/10, AUC_HF/10, SD1*1e3, SD2*1e3, Tot_HRV*1e6]
    data_df = {}
    for i, dv in enumerate(res_list):
        data_df[dv] = [res_tmp[i]]

    hrv_metrics_homemade = pd.DataFrame(data=data_df)

    #### for figures

    #### dHR
    if fig_token:
        fig_verif, fig_dHR = get_dHR(RRI_resample, srate_resample_hrv, f_RRI)

    #### fig
    if fig_token:
        fig_RRI = get_fig_RRI_IFR(ecg_i, ecg_cR, RRI, IFR, srate, srate_resample_hrv)
        fig_PSD = get_fig_PSD_LF_HF(Pxx, hzPxx, VLF, LF, HF) 
        fig_poincarre = get_fig_poincarre(RRI)

        fig_list = [fig_RRI, fig_PSD, fig_poincarre, fig_verif, fig_dHR]

        return hrv_metrics_homemade, fig_list

    else:

        return hrv_metrics_homemade








