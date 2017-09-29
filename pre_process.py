# Spike triggered average - as initialization or else

import numpy as np
import scipy.stats as sps
from scipy import signal

def STA(X,#X - stimuli, stimulus_size x number_of_data
        Y,#Y - responses, number_of_neurons x number_of_data
        crop,#width/heigth of cropped center bit where neurons are located
        smooth=0):#swidth/heigth of smoothing gaussian, if 0, no smoothing
    
    #sta - spike triggered average, number_of_neurons x stimulus_size
    s=np.sqrt(X.shape[0]).astype(int)
    d=X.shape[1]
    Y=Y.reshape([-1,d])
    n=Y.shape[0]
    
    sta = ((X.dot(Y.T))/Y.shape[1]).T
    sta = sta.reshape([n,s*s])
    #sta = sta**2
    
    #Smoothing
    if smooth>0:
        #create Gaussian
        x = np.linspace(1, smooth, smooth)
        y = np.linspace(smooth, 1, smooth)
        xm, ym = np.meshgrid(x, y)
        centre = [smooth/2+.5, smooth/2+.5]
        ind_tmp = (np.abs(xm-centre[0]) < smooth) & (np.abs(ym-centre[1]) < smooth)
        rf_tmp = np.zeros((smooth,smooth))
        rf_tmp[ind_tmp] = np.sqrt( (centre[0] - xm[ind_tmp])**2 +
                          (centre[1] - ym[ind_tmp])**2 )
        rf_tmp[ind_tmp] = (sps.norm.pdf(rf_tmp[ind_tmp],0,smooth**(1/4)))
        normal=rf_tmp
        #smooth
        sta_smooth=np.zeros(sta.shape)
        for i in range(n):
            sta_smooth[i,:] = signal.convolve2d(sta[i,:].reshape([s,s]),
                normal,mode='same').reshape(s**2)
    
    
    #cropping
    ind = np.int((s-crop)/2)
    sta = sta.reshape([n,s,s])
    sta = sta[:,ind:s-ind,ind:s-ind]
    sta = sta.reshape([n,crop**2])
    sta_smooth = sta_smooth.reshape([n,s,s])
    sta_smooth = sta_smooth[:,ind:s-ind,ind:s-ind]
    sta_smooth = sta_smooth.reshape([n,crop**2])
        
    return sta, sta_smooth



# Finding SD of noiseless signal (not on test set!)
def noise_cancel(RF,#neurons receptive fields
                 D,#calculate from how much data?
                 rep=30,#Lehky92 average responses over 30 repeats
                 np_seed=None):
    
    #set np random seed
    np.random.seed(np_seed)
    
    X = np.random.normal(0,1,[RF.shape[1],D])
    GT = np.dot(RF,X)
    Y=np.zeros([RF.shape[0],D])
    for r in range(rep):
        Y += GT + np.random.normal(0,np.sqrt(np.abs(GT)),GT.shape)
    Y /= rep
    SD = np.std(Y,axis=1)
    return SD# N x 1