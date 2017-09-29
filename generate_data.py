#generate random data on GPU (faster for large amounts)

import numpy as np
# Tensorflow needs to be LAST import
import tensorflow as tf

def generate_data(RF,# neurons receptive fields (from generate_neurons, where mean response is already set to 0.1)
                  num_train,
                  num_val,
                  tf_seed=None,#random seed
                  num_test=10**4,
                  noise=True):# whether to use poisson like noise on data
    
    #OUTPUTS:
    #Y_train - N(neurons) x num_train
    #Y_val - N x num_val
    #GT_test - N x num_test
    #X_train - s*s(input size) x num_train
    #X_val - s*s x num_val
    #X_test - s*s x num_test
    
    print('start generating data')
    
    num_pixel = RF.shape[1]
    rf=RF.astype(np.float32)
    
    #The following value depends on the size of the GPU memory:
    Max=10**5
    
    #allocate GPU memory dynamically?
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9999)
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth=True
    
    #Preallocate
    input_train1=np.zeros([num_pixel,num_train])
    output_train1=np.zeros([rf.shape[0],num_train])
    
    ##if too large for GPU memory
    if num_train>Max:
        D_all=np.hstack([np.tile(Max,num_train//Max),num_train%Max])
    else:
        D_all=[num_train]
        
    #First round do 1 complete block of size Max
    num_train = D_all[0]
    with tf.Graph().as_default():
        #set random seed
        tf.set_random_seed(tf_seed)
        with tf.device('/gpu:0'):
            # Input data
            input_train = tf.get_variable(name='input_train',shape=[num_pixel,num_train],
                                          initializer=tf.random_normal_initializer(),
                                          trainable=False,collections=[])
            input_val = tf.get_variable(name='input_val',shape=[num_pixel,num_val],
                                          initializer=tf.random_normal_initializer(),
                                          trainable=False,collections=[])
            input_test = tf.get_variable(name='input_test',shape=[num_pixel,num_test],
                                          initializer=tf.random_normal_initializer(),
                                          trainable=False,collections=[])
            # Ground truth receptive fields
            RF = tf.constant(rf)
            # Output data
            output_train = tf.matmul(RF,input_train)
            output_val = tf.matmul(RF,input_val)
            output_test = tf.matmul(RF,input_test)#this is ground truth
            # Add noise
            if noise:
                #(tested: 1.this really gives different SD for every draw, 2. += adds up)
                output_train += tf.random_normal(tf.shape(output_train),
                                               0,tf.sqrt(tf.abs(output_train)))
                output_val += tf.random_normal(tf.shape(output_val),
                                                   0,tf.sqrt(tf.abs(output_val)))

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            sess.run([input_train.initializer,input_val.initializer,input_test.initializer])
            output_train1[:,:num_train]=sess.run(output_train)
            input_train1[:,:num_train]=sess.run(input_train)
            input_val1=sess.run(input_val)
            input_test1=sess.run(input_test)
            output_val1=sess.run(output_val)
            output_test1=sess.run(output_test)
                
    #Other rounds, do missing training samples
    for i in range(1,len(D_all)):
        print('data generation %s finished'%(i/len(D_all)))
        num_train = D_all[i]
        
        with tf.Graph().as_default():
            #set random seed
            tf.set_random_seed(tf_seed)
            with tf.device('/gpu:0'):
                input_train = tf.get_variable(name='input_train',shape=[num_pixel,num_train],
                                              initializer=tf.random_normal_initializer(),
                                              trainable=False,collections=[])
                RF = tf.constant(rf)
                output_train = tf.matmul(RF,input_train)
                output_train += tf.random_normal(tf.shape(output_train),
                                               0,tf.sqrt(tf.abs(output_train)))
            
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                
                sess.run(input_train.initializer)
                output_train1[:,sum(D_all[:i]):sum(D_all[:i+1])]=sess.run(output_train)
                input_train1[:,sum(D_all[:i]):sum(D_all[:i+1])]=sess.run(input_train)

    return input_train1,input_val1,input_test1,output_train1,output_val1,output_test1



# when data is loaded, split data for cross val (define val set later, e.g. with k-fold)
def split_data(X,#images
               Y,#outputs
               num_test=10**4,
               noise=True,#use poisson like noise?
               np_seed=None,
               mean_response=.1):# mean response rate
    
    #set np random seed
    np.random.seed(np_seed)
    
    D=X.shape[0]#number of stimuli
    N=Y.shape[0]#number of neurons
    
    num_train = D-num_test
    
    X_train=X[:num_train,:,:,:]
    X_test=X[num_train:,:,:,:]
    
    Y_train=np.zeros([N,num_train])
    GT_test=np.zeros([N,num_test])
    
    for n in range(N):
        tmp_mean = np.mean(Y[n,:])
        Y_train[n,:] = Y[n,:num_train] / tmp_mean * mean_response
        GT_test[n,:] = Y[n,num_train:] / tmp_mean * mean_response
    
    #Poisson-like noise
    if noise:
        Y_train += np.random.normal(0,np.sqrt(np.abs(Y_train)),Y_train.shape)
    
    #Real Poisson noise
    #if noise:
    #    Y_train += np.random.poisson(Y_train)
    #    Y_test =  GT_test + np.random.poisson(GT_test)
    
    return Y_train,X_train,X_test,GT_test



# same as generate_data() above but on CPU
def generate_data_cpu(RF,
                      num_train,
                      num_val,
                      num_test=10**4,
                      noise=True,
                      np_seed=None):
    
    #set np random seed
    np.random.seed(np_seed)
    
    #preallocate
    N,size=RF.shape#Neurons, image width=heigth
    
    #white noise stimuli
    X_train = np.random.normal(0,1,[size,num_train])
    X_val = np.random.normal(0,1,[size,num_val])
    X_test = np.random.normal(0,1,[size,num_test])
    
    #Clean Responses
    Y_train = np.dot(RF,X_train)
    Y_val = np.dot(RF,X_val)
    GT_test = np.dot(RF,X_test)#ground truth without added noise
    
    #Poisson-like noise
    if noise:
        Y_train += np.random.normal(0,np.sqrt(np.abs(Y_train)),[N,train_num])
        Y_val +=  np.random.normal(0,np.sqrt(np.abs(Y_val)),[N,val_num])
        Y_test =  GT_test + np.random.normal(0,np.sqrt(np.abs(GT_test)),[N,test_num])
    
    return Y_train,Y_val,Y_test,X_train,X_val,X_test,GT_test