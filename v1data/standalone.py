import numpy as np
import tensorflow as tf
import os
import inspect
from scipy import stats
from convnet import ConvNet
from data import Dataset, load_data


def fit(data, filter_sizes, out_channels, strides, paddings,
        smooth_weights, sparse_weights, readout_sparse_weight,
        learning_rate=0.001, max_iter=10000, val_steps=50,
        early_stopping_steps=10):
    '''Fit CNN model.
    
    Parameters:
        data:                  Dataset object (see load_data())
        filter_sizes:          Filter sizes (list containing one number per conv layer)
        out_channels:          Number of output channels (list; one number per conv layer)
        strides:               Strides (list; one number per conv layer)
        paddings:              Paddings (list; one number per conv layer; VALID|SAME)
        smooth_weights:        Weights for smoothness regularizer (list; one number per conv layer)
        sparse_weights:        Weights for group sparsity regularizer (list; one number per conv layer)
        readout_sparse_weight: Sparisty of readout weights (scalar)
        learning_rate:         Learning rate (default: 0.001)   
        max_iter:              Max. number of iterations (default: 10000)
        val_steps:             Validation interval (number of iterations; default: 50)
        early_stopping_steps:  Tolerance for early stopping. Will stop optimizing 
            after this number of validation steps without decrease of loss.

    Output:
        cnn:                   A fitted ConvNet object
    '''
    cnn = ConvNet(data, log_dir='cnn', log_hash='manual')
    cnn.build(filter_sizes=filter_sizes,
              out_channels=out_channels,
              strides=strides,
              paddings=paddings,
              smooth_weights=smooth_weights,
              sparse_weights=sparse_weights,
              readout_sparse_weight=readout_sparse_weight)
    for lr_decay in range(3):
        training = cnn.train(max_iter=max_iter,
                             val_steps=val_steps,
                             early_stopping_steps=early_stopping_steps,
                             learning_rate=learning_rate)
        for (i, (logl, readout_sparse, conv_sparse, smooth, total_loss, pred)) in training:
            print('Step %d | Loss: %.2f | Poisson: %.2f | L1 readout: %.2f | Sparse: %.2f | Smooth: %.2f | Var(y): %.3f' % \
                  (i, total_loss, logl, readout_sparse, conv_sparse, smooth, np.mean(np.var(pred, axis=0))))
        learning_rate /= 3
        print('Reducing learning rate to %f' % learning_rate)
    print('Done fitting')
    return cnn
    

def evaluate(net):
    '''Evaluate CNN model.
    
    Parameters:
        net: A fitted ConvNet object
    
    Outputs:
        results: 
            A dictionary containing the following evaluation metrics:
                train_loss: Training loss
                val_loss:   Validation loss
            
            The following statistics are all evaluated on the test set:
                mse:        Mean-squared error for each cell
                avg_mse:    Average mean-squared error across cells
                corr:       Correlation between prediction and observation for each cell
                avg_corr:   Average correlation across cells
                var:        Variance of prediction for each cell
                avg_var:    Average variance across cells
                ve:         Variance explained for each cell
                avg_ve:     Average variance explained across cells
                eve:        Explainable variance explained for each cell
                    (Excludes an estimate of the observation noise. Be careful: this
                    quantity is not very reliable and needs to be taken with a grain
                    of salt)
                avg_eve:    Average explainable variance explained across cells
                nnp:        Normalized noise power (see Antolik et al. 2016)
    '''
    results = dict()
    train_loss = 0
    batch_size = 288
    net.data.next_epoch()
    for i in range(5):
        imgs_train, responses_train = net.data.minibatch(batch_size)
        train_loss += net.eval(images=imgs_train, responses=responses_train)[0]
    results['train_loss'] = train_loss / 5
    imgs_val, responses_val = net.data.val()
    results['val_loss'] = net.eval(images=imgs_val, responses=responses_val)[0]
    imgs_test, responses_test = net.data.test(averages=False)
    responses_test_avg = responses_test.mean(axis=0)
    result = net.eval(images=imgs_test, responses=responses_test_avg)
    results['mse'] = np.mean((result[-1] - responses_test_avg) ** 2, axis=0)
    results['avg_mse'] = results['mse'].mean()
    results['corr'] = np.array([stats.pearsonr(yhat, y)[0] if np.std(yhat) > 1e-5 and np.std(y) > 1e-5 else 0 for yhat, y in zip(result[-1].T, responses_test_avg.T)])
    results['avg_corr'] = results['corr'].mean()
    results['var'] = result[-1].var(axis=0)
    results['avg_var'] = results['var'].mean()
    results['ve'] = 1 - results['mse'] / responses_test_avg.var(axis=0)
    results['avg_ve'] = results['ve'].mean()
    reps, _, num_neurons = responses_test.shape
    obs_var_avg = (responses_test.var(axis=0, ddof=1) / reps).mean(axis=0)
    total_var_avg = responses_test.mean(axis=0).var(axis=0, ddof=1)
    results['eve'] = (total_var_avg - results['mse']) / (total_var_avg - obs_var_avg)
    results['avg_eve'] = results['eve'].mean()
    obs_var = (responses_test.var(axis=0, ddof=1)).mean(axis=0)
    total_var = responses_test.reshape([-1, num_neurons]).var(axis=0, ddof=1)
    results['nnp'] = obs_var / total_var
    return results

    
def main():
    print('Fitting CNN to data from region 1')
    data = load_data(region_num=1)
    cnn = fit(data,
              filter_sizes=[13, 3, 3],
              out_channels=[48, 48, 48],
              strides=[1, 1, 1],
              paddings=['VALID', 'SAME', 'SAME'],
              smooth_weights=[0.03, 0, 0],
              sparse_weights=[0, 0.05, 0.05],
              readout_sparse_weight=0.02)
    results = evaluate(cnn)
    print('Training loss: {:.2f} | Validation loss: {:.2f} | Average correlation on test set: {:.2f}'.format(
        results['train_loss'], results['val_loss'], results['avg_corr']))
        
        
if __name__ == "__main__":
    main()
