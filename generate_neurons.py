## Generate On-Center/Off-Sourround receptive fields

import numpy as np
import scipy.stats as sps
from scipy.stats import chi
import matplotlib.pyplot as plt

def generate_neurons(N,#number of neurons
                     s=48,#width=heigth of stimulus
                     s2=17,#width=heigth of Receptive fields
                     centering=True,#whether to center neuron on a pixel (or between)
                     target_mean_count=.1,#mean output
                     variation=0,#between neurons of same type
                     types=1,#adjust the mean of the type
                     np_seed=None):#Numpy seed

    #set np random seed
    np.random.seed(np_seed)
    
    #Output
    #rf - Receptive fields of N neurons
    #rf_cen - true kernel that generated them
    #GT_loc - true location
    
    #vary the following parameters by +/- 10%:
    #center size, surround size, center weight, surround weight, aspect ratio
    
    #from bc paper (franke et al,2017)
    #width: max diff between types: 1.5629, max var within type: 0.2341
    #heigth: max diff between types: 1.5044, max var within type: 0.2275
    
    s2=np.int((s2-1)/2)
    rf=np.zeros([N,s**2])
    rf_cen=np.zeros([N,(2*s2+1)**2])
    GT_loc=np.zeros([N,2])
    RF = np.zeros((N,s*s))#neurons receptive fields
    for n in range(N):
        if n%100==0:
            print('generate neurons %s finished'%(n/N))
        x2 = np.linspace(-s2, s2, 2*s2+1)
        y2 = np.linspace(-s2, s2, 2*s2+1)
        tmp_rf=np.zeros([s2*2+1,s2*2+1])

        #aspect ratio
        ar=np.random.uniform(1-variation,1+variation)
        #Center size
        c11=ar*np.random.uniform(1-variation,1+variation)/4*types
        c22=c11/ar**2
        #surround size
        s11=ar*np.random.uniform(1-variation,1+variation)*types
        s22=s11/ar**2
        #rotations - if variation is desired
        if variation != 0:
            tmp1=np.random.uniform(-.3,.3)
        else:
            tmp1=0
        tmp2=np.random.choice([0,1],2,replace=False)
        c12=[tmp1*np.min([c11,c22]),-tmp1*np.min([c11,c22])][tmp2[0]]
        c21=[tmp1*np.min([c11,c22]),-tmp1*np.min([c11,c22])][tmp2[1]]
        s12=[tmp1*np.min([s11,s22]),-tmp1*np.min([s11,s22])][tmp2[0]]
        s21=[tmp1*np.min([s11,s22]),-tmp1*np.min([s11,s22])][tmp2[1]]
        #center weight
        cw=np.random.uniform(1-variation,1+variation)
        #surround weight
        sw=cw*2#np.random.uniform(1-variation,1+variation)*2
        #covariance matrices
        ccov=s2*np.array([[c11,c12],[c21,c22]])
        scov=s2*np.array([[s11,s12],[s21,s22]])
        #not centering on pixels?
        if centering:
            center=[0,0]
        else:
            center=np.random.uniform(-.5,.5,2)

        #DoGs...
        for x in x2:
            for y in y2:
                tmp_rf[np.int(x)+s2,np.int(y)+s2]=cw*(sps.multivariate_normal.pdf([x,y],
                center,ccov) - sw*sps.multivariate_normal.pdf([x,y],center,scov))

        #shifting in space...
        tmp_cen=tmp_rf
        raw_mean_count = chi.mean(1, scale=np.sqrt(np.sum(tmp_cen ** 2)))
        tmp_cen *= target_mean_count / raw_mean_count
        rf_cen[n,:]=tmp_cen.flatten()

        centre=np.random.choice(s-2*s2,2)
        tmp_rf=np.vstack([np.zeros([centre[0],2*s2+1]),tmp_rf,np.zeros([s-2*s2-1-centre[0],2*s2+1])])
        tmp_rf=np.hstack([np.zeros([s,centre[1]]),tmp_rf,np.zeros([s,s-2*s2-1-centre[1]])])
        GT_loc[n,:]=centre+center#coordinates of true location

        #normalize all by the same amount(i.e. any neuron when RF is fully inside):
        raw_mean_count = chi.mean(1, scale=np.sqrt(np.sum(tmp_rf ** 2)))
        tmp_rf = tmp_rf * target_mean_count / raw_mean_count

        rf[n,:]=tmp_rf.flatten()
    
    return rf,rf_cen,GT_loc

def visualize(rf,show=3):
    s=np.int(np.sqrt(rf.shape[1]))

    for i in range(show**2):
        plt.subplot(show,show,i+1)
        plt.imshow(rf[i,:].reshape(s,s),cmap='bwr',vmin=-max(abs(rf[i,:])), vmax=max(abs(rf[i,:])))
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    plt.show()