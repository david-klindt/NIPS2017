## GLMs

import numpy as np
from sklearn import linear_model

##OLS without regularization
def ols(X,#X stimuli (s x D)
        Y):#Y responses (N x D)
    
    #OUT: RF_est estimated receptive fields (N x s)
    
    N=Y.shape[0]
    RF_est = np.zeros([Y.shape[0],X.shape[0]])
    #Fit each individually
    tmp_pinv=np.linalg.pinv(X.dot(X.T))# use pseudoinverse
    for n in range(N):
        RF_est[n,:] = tmp_pinv.dot(X.dot(Y[n,:]))

    return RF_est


#OLS with L1 reg, ie Lasso (Laplace prior)
#Needs cross-val over reg parameter!
def lasso(reg,#regularisation
          X,#X stimuli (s x D)
          Y):#Y responses (N x D)
    #OUT: RF_est estimated receptive fields (N x s)

    N=Y.shape[0]
    RF_est = np.zeros([Y.shape[0],X.shape[0]])
    
    #Fit each individually
    for n in range(N):
        las = linear_model.Lasso(reg)
        las.fit(X.T,Y[n,:])
        RF_est[n,:] = las.coef_

    return RF_est


##OLS with L2 reg, ie Ridge (Gauss prior)
#Needs cross-val over reg parameter
def ridge(reg,#regularisation
          X,#X stimuli (s x D)
          Y):#Y responses (N x D)
    #OUT: RF_est estimated receptive fields (N x s)

    N=Y.shape[0]
    RF_est = np.zeros([Y.shape[0],X.shape[0]])
    
    #Fit each individually
    for n in range(N):
        rid = linear_model.Ridge(reg)
        rid.fit(X.T,Y[n,:])
        RF_est[n,:] = rid.coef_

    return RF_est