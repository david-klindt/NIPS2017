# Training Schedule

import numpy as np
import CNN
import os
import matplotlib.pyplot as plt
# Tensorflow needs to be LAST import
import tensorflow as tf

def train(s,#        sizes of [image(heigth=width), image(depth=channel), mask]
          sK,#        s - sizes of [kernel1, kernel2, ...] - length of list gives number of layers
          act_fn,#      activation functions for each kernel = 'ID' or 'relu'
          init_scales,# rows: kernel,mask,weights; columns: mean, sd
          init_lr,# initial learning rate, goes down later
          num_kern,#        num_kern - number of kernels per conv layer
          max_runs,# maximum number of training steps
          reg,#        regularization weights [Kernel,Mask,Weights]
          batch_size,# (max gpu memory?)256 for small one layer, 64 for large multilayer model
          X_train,X_val,X_test,Y_train,Y_val,GT_test,#Data, X in NCHW format, Y: Neurons x Data
          kernel_constraint=None,#constraint for kernel weights - 'norm' == 1
          weights_constraint=None,#constraint on read out weights - 'abs','norm','absnorm'
          mask_constraint=None,#constraint on mask - 'abs'
          final_relu=False,
          stop_crit=[5,3],#[after how many worse steps lower lr, lower how many times]
          burn_in=0,#allow initial exploration of 100*burn_in runs 
          types=[0],#starting indices of types, default=all same type
          GT_mask=np.array([]),#true locations if known: rows:neurons, columns: x,y location
          tf_seed=None,#random seed for tensorflow
          np_seed=None,
          split_data=True,#whether to go through test and val set in batch_size chunks
          sta=None,#if provided, take maximum absolute pixel of sta to initialize mask
          sd=np.array([]),#standard deviation of responses, to set the scale
          GT_WK=None,#initialize first conv kernel?
          batch_norm=True,#whether to use batch norm
          bn_cent=False,#center after batch norm?
          verbose='minimal'):#print training outcome and learned filters every 100 steps, 'yes','no','minimal'
    
        ##Visualization function
    def visualization():
        #display conv kernels if only one conv layer and 1 input channel
        if len(num_kern)==1 and s[1]:
            K=num_kern[0]
        else:
            K=0
        
        #to calculate FEV and show predicted responses
        if split_data:
            test=np.zeros(num_test)
            for i in range(num_test):
                test[i]=sess.run(model.MSE,feed_test[i])
            MSE_gt = np.mean(test)#mean squared errors
            test_y=sess.run(model.Y_,feed_test[0]).T#responses
        else:
            MSE_gt = sess.run(model.MSE,feed_test)
            test_y=sess.run(model.Y_,feed_test).reshape([N,-1])[:,:batch_size].T
        print('Total FEV = ',(1 - MSE_gt/np.mean(gt_test_var)))
        if j:#only if it ran already
            print('Runs: %s; MSE - train: %s, val: %s; lr = %s'%
              (j,MSE_train[-1],MSE_val[-1],lr))
            print('best run: ',(tmp_min_ind+1)*100,MSE_val[tmp_min_ind])
        #Regularization losses:
        print('Loss/Regularization: %s MSE, %s kernel, %s mask, %s weights'%(
              MSE_gt,model.regK.eval()*reg[0],
              model.regM.eval()*reg[1],model.regW.eval()*reg[2]))
        
        #plot
        fig, ax = plt.subplots(1, 4+K, figsize=[18, 3])
        #Training progress
        if j:
            ax[0].plot(MSE_val)
            ax[0].plot(MSE_train)
            #ax[0].set_ylim([min(MSE_train),2*np.median(MSE_val)-min(MSE_train)])
        ax[0].legend(['MSE Val','MSE Train'])
        #Predicted vs true
        ax[1].plot(GT_test[:,:batch_size].T, test_y, '.')
        xx = [-.4, .4]
        ax[1].plot(xx, xx)
        ax[1].axis('equal')
        ax[1].set_title('true vs predicted')
        #Example mask
        tmp_wm=model.WM.eval()[:,0].T#first neuron as example
        ax[2].imshow(tmp_wm.reshape([s[2],s[2]]),cmap='bwr',
            vmin=-max(abs(tmp_wm.T)),vmax=max(abs(tmp_wm.T)))
        ax[2].get_xaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)
        ax[2].set_title('Mask N_1')
        if GT_mask.size:#if true location is known
            ax[2].axhline(y=GT_mask[0,0], xmin=0, xmax=s[2], linewidth=1, color ='g')
            ax[2].axvline(x=GT_mask[0,1], ymin=0, ymax =s[2], linewidth=1, color='g')
        #All Weights
        if num_kern[-1]>1:
            tmp_ww=model.WW.eval()
            for t in range(len(types)-1):
                if num_kern[-1]==1:
                    ax[3].plot(np.arange(types[t],types[t+1]),tmp_ww[:,types[t]:types[t+1]].T,'o')
                else:
                    ax[3].plot(tmp_ww[:,types[t]:types[t+1]],color=plt.cm.gist_rainbow(t/len(types)))
            ax[3].set_title('All Weights')
        #Conv Kernels
        if K>0:
            tmp_wk=model.WK[0].eval().reshape([sK[0],sK[0],K])
            for k in range(K):
                ax[4+k].imshow(tmp_wk[:,:,k],cmap='bwr',
                        vmin=-np.max(abs(tmp_wk)),vmax=np.max(abs(tmp_wk)))
                ax[4+k].get_xaxis().set_visible(False)
                ax[4+k].get_yaxis().set_visible(False)
                ax[4+k].set_title('Kernel%s'%(k+1))

        plt.show()
        return
    
    #set np random seed
    np.random.seed(np_seed)
    
    #Derive parameters
    N=Y_train.shape[0]
    types.append(N)#add last index
    
    #Storing:
    MSE_train = [] # MSE on train set
    MSE_val = []#MSE on validation set
    MSE_test = []#MSE on test
    WK = []# Kernel - store best weights
    WM = []# Mask - store best weights
    WW = []# Read Out Weights - store best weights
    FEV = []#fraction of explained variance, 1 x 1

    #calculate test variance
    gt_test_var = np.var(GT_test,axis =1)#explainable output variance per cell
    
    #initialize learning parameters
    lr=init_lr
    j=None#run index
    # flags for early stopping
    stop_flagA = 0#decrease learning rate
    stop_flagB = 0#final stop

    #Init Mask weights
    tmp = np.random.normal(0,init_scales[1,1],sta.shape)
    if sta.size:
        #STA - maximum pixel (use the smoothed STA?, better estimation?)
        if sd.size:
            tmp[np.arange(N),np.argmax(abs(sta),1)] = sd
        else:
            tmp[np.arange(N),np.argmax(abs(sta),1)] = np.ones(N)*init_scales[1,0]
            #ground truth masks:
            #tmp[np.arange(N),(np.round(GT_mask[:,0])*32+np.round(GT_mask[:,1])
            #    ).astype(int)] = np.ones(N)*init_scales[1,0]
    init_mask = tmp.astype(np.float32).T
    
    #init Weights
    init_weights = np.random.normal(init_scales[2,0],init_scales[2,1],[num_kern[-1],N])
    
    #Init model class
    model = CNN.ModelGraph(s,sK,reg,init_scales,N,num_kern,act_fn,kernel_constraint,weights_constraint,
                           mask_constraint=mask_constraint,final_relu=final_relu,np_seed=np_seed,
                           batch_norm=batch_norm,bn_cent=bn_cent,tf_seed=tf_seed,init_mask=init_mask,
                           init_weights=init_weights)#,
    #                       init_kernel=GT_WK.astype(np.float32))

    #validation and test feed can be outside loop:
    if split_data:
        #Split up if too large for GPU memory...
        num_test=np.int(GT_test.shape[1]/batch_size)
        feed_test=[]
        for i in range(num_test):
            feed_test.append( {model.X:X_test[i*batch_size:(i+1)*batch_size,:,:,:],
                               model.Y:GT_test[:,i*batch_size:(i+1)*batch_size],
                               model.is_train:False})
        num_val=np.int(Y_val.shape[1]/batch_size)
        feed_val=[]
        for i in range(num_val):
            feed_val.append( {model.X:X_val[i*batch_size:(i+1)*batch_size,:,:,:],
                              model.Y:Y_val[:,i*batch_size:(i+1)*batch_size],
                              model.is_train:False})
    else:
        feed_val = {model.X:X_val,model.Y:Y_val,model.is_train:False}
        feed_test = {model.X:X_test,model.Y:GT_test,model.is_train:False}
    
    
    #Clean previous checkpoints
    files = os.listdir()
    for file in files:
        if file.startswith("bn_checkpoint"):
            os.remove(file)
    
    ##Start a tf session
    with model.graph.as_default():
        #set random seed
        tf.set_random_seed(tf_seed)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            #plot initial weights
            if verbose=='yes':
                print('Before Training:')
                visualization()

            #Batches - define list of starting-indices for individual batches in data set:
            #if there is less training data than batch size
            batch_size = np.min([batch_size,X_train.shape[0]])
            batch_ind = np.arange(0,X_train.shape[0],batch_size)
            #number of selected batch
            batch = 0

            #Optimization runs
            for j in range(1,max_runs):

                #when there is no further complete batch
                if batch==len(batch_ind):
                    #shuffle data and start again:
                    ind = np.random.permutation(X_train.shape[0])
                    X_train = X_train[ind,:,:,:]
                    Y_train = Y_train[:,ind]
                    batch = 0

                #take a batch
                X_batch = X_train[batch_ind[batch]:batch_ind[batch]+batch_size,:,:,:]
                Y_batch = Y_train[:,batch_ind[batch]:batch_ind[batch]+batch_size]
                batch +=1
                
                #Training feed:
                feed_dict ={model.step_size:lr,model.X:X_batch,
                            model.Y:Y_batch,model.is_train:True}

                # Training with current batch:
                sess.run([model.training, model.update_ops],feed_dict)
                
                #Early Stopping - check if MSE doesn't increase
                if j%100==0:
                    #save current parameters
                    model.saver.save(sess, 'bn_checkpoint', global_step=int(j/100))

                    # Store MSE on train:
                    MSE_train.append(model.MSE.eval(feed_dict))

                    #check MSE on validation set and store the parameters
                    if split_data:
                        val=np.zeros(num_val)
                        for i in range(num_val):
                            val[i]=sess.run(model.MSE,feed_val[i])
                        MSE_val.append(np.mean(val))
                    else:
                        MSE_val.append(model.MSE.eval(feed_val))

                    #Best run
                    tmp_min_ind = np.argmin(MSE_val)

                    ##Display progress?
                    if verbose=='yes':
                        visualization()
                    elif verbose=='minimal':
                        print('run = %s, MSE_val = %s, MSE_train = %s'%(j,MSE_val[-1],MSE_train[-1]))

                    #Early Stopping - if latest validation MSE is not minimum
                    if (tmp_min_ind != len(MSE_val)-1) and (len(MSE_val)>burn_in):
                        stop_flagA +=1
                        if stop_flagA>=stop_crit[0]:#how many steps worse than best, before lowering lr?
                            lr *= .1
                            stop_flagA = 0
                            stop_flagB +=1
                            if stop_flagB==stop_crit[1]:#lower the lr x times
                                break
                    else:#if latest value is best, reset
                        stop_flagA = 0

            #Best run
            tmp_min_ind = np.argmin(MSE_val)#len(MSE_val)-1#
            
            #Assign the best weights to model graph 
            model.saver.restore(sess, './bn_checkpoint-%s'%(tmp_min_ind+1))
            
            #Store best weights (i.e. lowest validation MSE)
            for k in range(len(num_kern)):
                WK.append(model.WK[k].eval())#List of HWInOut
            WM = model.WM.eval()#s[2] x N
            WW = model.WW.eval()#num_kern[-1] x N

            #clean checkpoints
            files = os.listdir()
            for file in files:
                if file.startswith("bn_checkpoint"):
                    os.remove(file)
                    
            #check MSE on validation set and store the parameters
            if split_data:
                val_Y_ = np.zeros([N,num_val*batch_size])
                for i in range(num_val):
                    val_Y_[:,i*batch_size:(i+1)*batch_size]=sess.run(model.Y_,feed_val[i])
            else:
                val_Y_ = model.Y_.eval(feed_val)
            Val_cell = np.mean((val_Y_-Y_val)**2,1)
                
            # Performance and predicted responses
            if split_data:
                Y_=np.zeros([N,num_test*batch_size])
                for i in range(num_test):
                    Y_[:,i*batch_size:(i+1)*batch_size]=sess.run(model.Y_,feed_test[i])
            else:
                Y_ = sess.run(model.Y_,feed_test)
            MSE_cell = np.mean((Y_-GT_test)**2,1)
                
            #reshape into matrix if N=1
            if N==1:
                Y_ = Y_.reshape([N,-1])
                MSE_cell = MSE_cell.reshape([N,-1])
                
            #FEV - fraction of explainable variance
            FEV_cell = 1 - (MSE_cell/gt_test_var)
            MSE_test = np.mean(MSE_cell)
            FEV = np.mean(FEV_cell)
            
            #Mean prediction (from mean)
            Y_mean = np.mean(Y_train,1).reshape([N,1])
            MSE_val_mean = np.mean((Y_mean*np.ones(Y_val.shape)-Y_val)**2,1)
            FEV_mean = 1 - (np.mean((Y_mean*np.ones(GT_test.shape)-GT_test)**2,1)/gt_test_var)

            #Output
            log=('Stop at run %s; MSE on validation set: %s'% (j,MSE_val[tmp_min_ind]),
                  'MSE on test set: %s; Mean FEV: %s' % (MSE_test, FEV))
            print(log)

    return (WK,WM,WW,MSE_train,MSE_val,Val_cell,MSE_test,FEV,FEV_cell,Y_,log,MSE_val_mean,FEV_mean)