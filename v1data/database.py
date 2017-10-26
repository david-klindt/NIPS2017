import datajoint as dj
import numpy as np
import os
import inspect
import convnet
from scipy import stats
from data import Dataset, load_data

schema = dj.schema('aecker_cnn_antolik', locals())


# constants
CONV_SMOOTH = [0.001, 0.003, 0.01]
CONV_SPARSE = [0, 0.025, 0.05]
READOUT_SPARSE = [0.03, 0.04, 0.05]
REG_GRID = [[a, b, c] for a in CONV_SMOOTH for b in CONV_SPARSE for c in READOUT_SPARSE]
REG_GRID_EXTENDED = [[0.01, b, c] for b in [0.025, 0.05] for c in [0.02]] + \
                    [[0.03, b, c] for b in [0.025, 0.05] for c in [0.02, 0.03]]

# constants for fully connected readout
ACT_SPARSE = [0.01, 0.02, 0.04]
REG_GRID_FC = [[a, b, c] for a in CONV_SMOOTH for b in CONV_SPARSE for c in ACT_SPARSE]


@schema
class Net(dj.Lookup):
    definition = """  # CNN definition
    net_id                  : int       # id
    """
    contents = [[i] for i in range(11)]


    class ConvLayer(dj.Part):
        definition = """  # CNN layer
        -> Net
        layer_num           : tinyint               # layer number
        ---
        filter_size         : tinyint               # filter size
        out_channels        : tinyint               # number of output channels (feature maps)
        stride              : tinyint               # stride
        padding             : enum('SAME', 'VALID') # type of padding
        rel_smooth_weight   : float                 # relative weight for smoothness regularizer
        rel_sparse_weight   : float                 # relative weight for sparseness regularizer
        """
        contents = [
            [0, 1, 13, 32, 1, 'VALID', 1, 1],

            [1, 1, 13, 16, 1, 'VALID', 1, 0],
            [1, 2, 3, 16, 1, 'SAME', 0, 1],

            [2, 1, 13, 32, 1, 'VALID', 1, 0],
            [2, 2, 3, 32, 1, 'SAME', 0, 1],

            [3, 1, 13, 48, 1, 'VALID', 1, 0],
            [3, 2, 3, 48, 1, 'SAME', 0, 1],

            [4, 1, 13, 16, 1, 'VALID', 1, 0],
            [4, 2, 3, 16, 1, 'SAME', 0, 1],
            [4, 3, 3, 16, 1, 'SAME', 0, 1],
            
            [5, 1, 13, 32, 1, 'VALID', 1, 0],
            [5, 2, 3, 32, 1, 'SAME', 0, 1],
            [5, 3, 3, 32, 1, 'SAME', 0, 1],
            
            [6, 1, 13, 48, 1, 'VALID', 1, 0],
            [6, 2, 3, 48, 1, 'SAME', 0, 1],
            [6, 3, 3, 48, 1, 'SAME', 0, 1],

            [7, 1, 13, 16, 1, 'VALID', 1, 0],
            [7, 2, 8, 16, 1, 'SAME', 1, 1],
            [7, 3, 8, 16, 1, 'SAME', 1, 1],
            
            [8, 1, 13, 32, 1, 'VALID', 1, 0],
            [8, 2, 8, 32, 1, 'SAME', 1, 1],
            [8, 3, 8, 32, 1, 'SAME', 1, 1],
            
            [9, 1, 13, 48, 1, 'VALID', 1, 0],
            [9, 2, 8, 48, 1, 'SAME', 1, 1],
            [9, 3, 8, 48, 1, 'SAME', 1, 1],
            
        ]


    class RegPath(dj.Part):
        definition = """  # Regularization path for each net
        -> Net
        reg_param_id                : smallint  # id
        ---
        conv_smooth_weight          : float     # weight for smoothness regularizer on conv layer
        conv_sparse_weight          : float     # weight for sparseness regularizer on conv layer
        readout_sparse_weight       : float     # weight for sparseness regularizer on readout layer
        """
        contents = [
            [0, j, a, b, c] for j, (a, b, c) in enumerate(
                [[a, b, c] for a in CONV_SMOOTH for b in [0.0] for c in READOUT_SPARSE])] + [
            [i, j, a, b, c] for i in range(1, 10) for j, (a, b, c) in enumerate(REG_GRID)] + [
            [i, j+len(REG_GRID), a, b, c] for i in range(5, 10) for j, (a, b, c) in enumerate(REG_GRID_EXTENDED)] + [
        ]


@schema
class Region(dj.Lookup):
    definition = """  # CNN fitting and results
    region_num              : tinyint   # region number
    """
    contents = [[i] for i in range(1, 4)]

    def load_data(self):
        # load the data
        assert len(self) == 1, 'relation must be scalar'
        region = self.fetch1()
        return load_data(region['region_num'])


def evaluate_model(net, tuple):
    tuple['log_hash'] = net.log_hash
    tuple['iterations'] = net.global_step
    train_loss = 0
    batch_size = 288
    net.data.next_epoch()
    for i in range(5):
        imgs_train, responses_train = net.data.minibatch(batch_size)
        train_loss += net.eval(images=imgs_train, responses=responses_train)[0]
    tuple['train_loss'] = train_loss / 5
    imgs_val, responses_val = net.data.val()
    tuple['val_loss'] = net.eval(images=imgs_val, responses=responses_val)[0]
    imgs_test, responses_test = net.data.test(averages=False)
    responses_test_avg = responses_test.mean(axis=0)
    result = net.eval(images=imgs_test, responses=responses_test_avg)
    tuple['mse'] = np.mean((result[-1] - responses_test_avg) ** 2, axis=0)
    tuple['avg_mse'] = tuple['mse'].mean()
    tuple['corr'] = np.array([stats.pearsonr(yhat, y)[0] if np.std(yhat) > 1e-5 and np.std(y) > 1e-5 else 0 for yhat, y in zip(result[-1].T, responses_test_avg.T)])
    tuple['avg_corr'] = tuple['corr'].mean()
    tuple['var'] = result[-1].var(axis=0)
    tuple['avg_var'] = tuple['var'].mean()
    tuple['ve'] = 1 - tuple['mse'] / responses_test_avg.var(axis=0)
    tuple['avg_ve'] = tuple['ve'].mean()
    reps, _, num_neurons = responses_test.shape
    obs_var_avg = (responses_test.var(axis=0, ddof=1) / reps).mean(axis=0)
    total_var_avg = responses_test.mean(axis=0).var(axis=0, ddof=1)
    tuple['eve'] = (total_var_avg - tuple['mse']) / (total_var_avg - obs_var_avg)
    tuple['avg_eve'] = tuple['eve'].mean()
    obs_var = (responses_test.var(axis=0, ddof=1)).mean(axis=0)
    total_var = responses_test.reshape([-1, num_neurons]).var(axis=0, ddof=1)
    tuple['nnp'] = obs_var / total_var
    return tuple


@schema
class Fit(dj.Computed):
    definition = """  # CNN fitting and results
    -> Region
    -> Net.RegPath
    ---
    log_hash                : varchar(10)   # hash for log directory
    iterations              : int           # numer of iterations
    train_loss              : float         # loss on training set
    val_loss                : float         # loss on validation set
    mse                     : blob          # MSE per cell on test set
    avg_mse                 : float         # average MSE on test set
    corr                    : blob          # correlation between prediction and response
    avg_corr                : float         # average correlation
    var                     : blob          # variance of prediction
    avg_var                 : float         # average variance
    ve                      : blob          # variance explained (1 - MSE/Var)
    avg_ve                  : float         # average VE
    eve                     : blob          # explainable variance explained (1 - MSE/ExplVar)
    avg_eve                 : float         # average EVE
    nnp                     : blob          # normalized noise power (noise var / total var)
    """

    def _make_tuples(self, key):

        cnn = Fit.build_net(key)
        learning_rate = 0.001
        for lr_decay in range(3):
            training = cnn.train(max_iter=10000,
                                 val_steps=50,
                                 early_stopping_steps=10,
                                 learning_rate=learning_rate)
            for (i, (logl, readout_sparse, conv_sparse, smooth, total_loss, pred)) in training:
                print('Step %d | Loss: %s | Poisson: %s | L1 readout: %s | Sparse: %s | Smooth: %s | Var(y): %s' % (i, total_loss, logl, readout_sparse, conv_sparse, smooth, np.mean(np.var(pred, axis=0))))
            learning_rate /= 3
            print('Reducing learning rate to %f' % learning_rate)
        print('Done fitting')

        tupl = evaluate_model(cnn, key)
        self.insert1(tupl)


    def get_net(self):
        assert len(self) == 1, 'Relation must be scalar!'
        log_hash, iterations = self.fetch1('log_hash', 'iterations')
        cnn = Fit.build_net(list(self.fetch.keys())[0], log_hash=log_hash, global_step=iterations)
        cnn.load()
        return cnn


    @staticmethod
    def build_net(key, log_hash=None, global_step=None):
        fit = Fit() & key
        data = (Region() & key).load_data()
        cnn = convnet.ConvNet(data, log_dir='region%d' % key['region_num'], log_hash=log_hash, global_step=global_step)
        layers = (Net.ConvLayer() & key).fetch(order_by='layer_num')
        conv_smooth, conv_sparse, readout_sparse = (Net.RegPath() & key).fetch1(
            'conv_smooth_weight', 'conv_sparse_weight', 'readout_sparse_weight')
        cnn.build(filter_sizes=layers['filter_size'],
                  out_channels=layers['out_channels'],
                  strides=layers['stride'],
                  paddings=layers['padding'],
                  smooth_weights=layers['rel_smooth_weight']*conv_smooth,
                  sparse_weights=layers['rel_sparse_weight']*conv_sparse,
                  readout_sparse_weight=readout_sparse)
        return cnn


@schema
class NetFC(dj.Lookup):
    definition = """  # CNN definition
    net_id                  : int       # id
    """
    contents = [[i] for i in range(11)]


    class ConvLayer(dj.Part):
        definition = """  # CNN layer
        -> NetFC
        layer_num           : tinyint               # layer number
        ---
        filter_size         : tinyint               # filter size
        out_channels        : tinyint               # number of output channels (feature maps)
        stride              : tinyint               # stride
        padding             : enum('SAME', 'VALID') # type of padding
        rel_smooth_weight   : float                 # relative weight for smoothness regularizer
        rel_sparse_weight   : float                 # relative weight for sparseness regularizer
        """
        contents = [
            [0, 1, 13, 16, 1, 'VALID', 1, 0],
            [0, 2, 3, 8, 1, 'SAME', 0, 1],
            [0, 3, 3, 4, 1, 'SAME', 0, 1],
            
            [1, 1, 13, 32, 1, 'VALID', 1, 0],
            [1, 2, 3, 16, 1, 'SAME', 0, 1],
            [1, 3, 3, 4, 1, 'SAME', 0, 1],
            
            [2, 1, 13, 32, 1, 'VALID', 1, 0],
            [2, 2, 3, 16, 1, 'SAME', 0, 1],
            [2, 3, 3, 8, 1, 'SAME', 0, 1],
            
            [3, 1, 13, 32, 1, 'VALID', 1, 0],
            [3, 2, 3, 32, 1, 'SAME', 0, 1],
            [3, 3, 3, 8, 1, 'SAME', 0, 1],
            
            [4, 1, 13, 32, 1, 'VALID', 1, 0],
            [4, 2, 3, 32, 1, 'SAME', 0, 1],
            [4, 3, 3, 16, 1, 'SAME', 0, 1],
            
            [5, 1, 13, 32, 1, 'VALID', 1, 0],
            [5, 2, 3, 32, 1, 'SAME', 0, 1],
            [5, 3, 3, 32, 1, 'SAME', 0, 1],

            [6, 1, 13, 48, 1, 'VALID', 1, 0],
            [6, 2, 3, 16, 1, 'SAME', 0, 1],
            [6, 3, 3, 4, 1, 'SAME', 0, 1],
            
            [7, 1, 13, 48, 1, 'VALID', 1, 0],
            [7, 2, 3, 16, 1, 'SAME', 0, 1],
            [7, 3, 3, 8, 1, 'SAME', 0, 1],
            
            [8, 1, 13, 48, 1, 'VALID', 1, 0],
            [8, 2, 3, 32, 1, 'SAME', 0, 1],
            [8, 3, 3, 4, 1, 'SAME', 0, 1],
            
            [9, 1, 13, 48, 1, 'VALID', 1, 0],
            [9, 2, 3, 32, 1, 'SAME', 0, 1],
            [9, 3, 3, 8, 1, 'SAME', 0, 1],
            
            [10, 1, 13, 32, 1, 'VALID', 1, 0],
            [10, 2, 3, 32, 1, 'SAME', 0, 1],
            [10, 3, 3, 4, 1, 'SAME', 0, 1],
        ]


    class RegPath(dj.Part):
        definition = """  # Regularization path for each net
        -> NetFC
        reg_param_id                : smallint  # id
        ---
        conv_smooth_weight          : float     # weight for smoothness regularizer on conv layer
        conv_sparse_weight          : float     # weight for sparseness regularizer on conv layer
        readout_sparse_weight       : float     # weight for sparseness regularizer on output layer
        """
        contents = [
            [i, j, a, b, c] for i in range(11) for j, (a, b, c) in enumerate(REG_GRID_FC)] + [
        ]

    
@schema
class FitFC(dj.Computed):
    definition = """  # CNN fitting and results
    -> Region
    -> NetFC.RegPath
    ---
    log_hash                : varchar(10)   # hash for log directory
    iterations              : int           # numer of iterations
    train_loss              : float         # loss on training set
    val_loss                : float         # loss on validation set
    mse                     : blob          # MSE per cell on test set
    avg_mse                 : float         # average MSE on test set
    corr                    : blob          # correlation between prediction and response
    avg_corr                : float         # average correlation
    var                     : blob          # variance of prediction
    avg_var                 : float         # average variance
    ve                      : blob          # variance explained (1 - MSE/Var)
    avg_ve                  : float         # average VE
    eve                     : blob          # explainable variance explained (1 - MSE/ExplVar)
    avg_eve                 : float         # average EVE
    nnp                     : blob          # normalized noise power (noise var / total var)
    """

    def _make_tuples(self, key):

        cnn = FitFC.build_net(key)
        learning_rate = 0.001
        for lr_decay in range(4):
            training = cnn.train(max_iter=10000,
                                 val_steps=50,
                                 early_stopping_steps=3,
                                 learning_rate=learning_rate)
            for (i, (logl, readout_sparse, conv_sparse, smooth, total_loss, pred)) in training:
                print('Step %d | Loss: %s | Poisson: %s | L1 readout: %s | Sparse: %s | Smooth: %s | Var(y): %s' % (i, total_loss, logl, readout_sparse, conv_sparse, smooth, np.mean(np.var(pred, axis=0))))
            learning_rate /= 10
            print('Reducing learning rate to %f' % learning_rate)
        print('Done fitting')

        tupl = evaluate_model(cnn, key)
        self.insert1(tupl)


    def get_net(self):
        assert len(self) == 1, 'Relation must be scalar!'
        log_hash, iterations = self.fetch1('log_hash', 'iterations')
        cnn = FitFC.build_net(list(self.fetch.keys())[0], log_hash=log_hash, global_step=iterations)
        cnn.load_best()
        return cnn


    @staticmethod
    def build_net(key, log_hash=None, global_step=None):
        fit = FitFC() & key
        data = (Region() & key).load_data()
        cnn = convnet.ConvNet(data, log_dir='fc', log_hash=log_hash, global_step=global_step)
        layers = (NetFC.ConvLayer() & key).fetch(order_by='layer_num')
        conv_smooth, conv_sparse, readout_sparse = (NetFC.RegPath() & key).fetch1(
            'conv_smooth_weight', 'conv_sparse_weight', 'readout_sparse_weight')
        cnn.build(filter_sizes=layers['filter_size'],
                  out_channels=layers['out_channels'],
                  strides=layers['stride'],
                  paddings=layers['padding'],
                  smooth_weights=layers['rel_smooth_weight']*conv_smooth,
                  sparse_weights=layers['rel_sparse_weight']*conv_sparse,
                  readout_sparse_weight=readout_sparse,
                  fully_connected_readout=True)
        return cnn



@schema
class NetFixedMask(dj.Lookup):
    definition = """  # CNN definition
    net_id                  : int       # id
    """
    contents = [[i] for i in range(11)]


    class ConvLayer(dj.Part):
        definition = """  # CNN layer
        -> NetFixedMask
        layer_num           : tinyint               # layer number
        ---
        filter_size         : tinyint               # filter size
        out_channels        : tinyint               # number of output channels (feature maps)
        stride              : tinyint               # stride
        padding             : enum('SAME', 'VALID') # type of padding
        rel_smooth_weight   : float                 # relative weight for smoothness regularizer
        rel_sparse_weight   : float                 # relative weight for sparseness regularizer
        """
        contents = [
            [0, 1, 13, 16, 1, 'VALID', 1, 0],
            [0, 2, 3, 16, 1, 'SAME', 0, 1],
            [0, 3, 3, 16, 1, 'SAME', 0, 1],
            
            [1, 1, 13, 32, 1, 'VALID', 1, 0],
            [1, 2, 3, 32, 1, 'SAME', 0, 1],
            [1, 3, 3, 32, 1, 'SAME', 0, 1],
            
            [2, 1, 13, 48, 1, 'VALID', 1, 0],
            [2, 2, 3, 48, 1, 'SAME', 0, 1],
            [2, 3, 3, 48, 1, 'SAME', 0, 1],
        ]


    class RegPath(dj.Part):
        definition = """  # Regularization path for each net
        -> NetFixedMask
        reg_param_id                : smallint  # id
        ---
        conv_smooth_weight          : float     # weight for smoothness regularizer on conv layer
        conv_sparse_weight          : float     # weight for sparseness regularizer on conv layer
        readout_sparse_weight       : float     # weight for sparseness regularizer on output layer
        """
        contents = [
            [i, j, a, b, c] for i in range(3) for j, (a, b, c) in enumerate(REG_GRID)] + [
        ]

    
@schema
class FitFixedMask(dj.Computed):
    definition = """  # CNN fitting and results
    -> Region
    -> NetFixedMask.RegPath
    ---
    log_hash                : varchar(10)   # hash for log directory
    iterations              : int           # numer of iterations
    train_loss              : float         # loss on training set
    val_loss                : float         # loss on validation set
    mse                     : blob          # MSE per cell on test set
    avg_mse                 : float         # average MSE on test set
    corr                    : blob          # correlation between prediction and response
    avg_corr                : float         # average correlation
    var                     : blob          # variance of prediction
    avg_var                 : float         # average variance
    ve                      : blob          # variance explained (1 - MSE/Var)
    avg_ve                  : float         # average VE
    eve                     : blob          # explainable variance explained (1 - MSE/ExplVar)
    avg_eve                 : float         # average EVE
    nnp                     : blob          # normalized noise power (noise var / total var)
    """

    def _make_tuples(self, key):

        cnn = FitFixedMask.build_net(key)
        learning_rate = 0.001
        for lr_decay in range(3):
            training = cnn.train(max_iter=10000,
                                 val_steps=50,
                                 early_stopping_steps=10,
                                 learning_rate=learning_rate)
            for (i, (logl, readout_sparse, conv_sparse, smooth, total_loss, pred)) in training:
                print('Step %d | Loss: %s | Poisson: %s | L1 readout: %s | Sparse: %s | Smooth: %s | Var(y): %s' % (i, total_loss, logl, readout_sparse, conv_sparse, smooth, np.mean(np.var(pred, axis=0))))
            learning_rate /= 3
            print('Reducing learning rate to %f' % learning_rate)
        print('Done fitting')

        tupl = evaluate_model(cnn, key)
        self.insert1(tupl)


    def get_net(self):
        assert len(self) == 1, 'Relation must be scalar!'
        log_hash, iterations = self.fetch1('log_hash', 'iterations')
        cnn = FitFixedMask.build_net(list(self.fetch.keys())[0], log_hash=log_hash, global_step=iterations)
        cnn.load_best()
        return cnn


    @staticmethod
    def build_net(key, log_hash=None, global_step=None):
        fit = FitFixedMask() & key
        data = (Region() & key).load_data()
        cnn = convnet.ConvNet(data, log_dir='fixed_mask', log_hash=log_hash, global_step=global_step)
        layers = (NetFixedMask.ConvLayer() & key).fetch(order_by='layer_num')
        conv_smooth, conv_sparse, readout_sparse = (NetFixedMask.RegPath() & key).fetch1(
            'conv_smooth_weight', 'conv_sparse_weight', 'readout_sparse_weight')
        cnn.build(filter_sizes=layers['filter_size'],
                  out_channels=layers['out_channels'],
                  strides=layers['stride'],
                  paddings=layers['padding'],
                  smooth_weights=layers['rel_smooth_weight']*conv_smooth,
                  sparse_weights=layers['rel_sparse_weight']*conv_sparse,
                  readout_sparse_weight=readout_sparse,
                  fixed_rfs=True)
        return cnn



@schema
class LnpRegPath(dj.Lookup):
    definition = """  # Regularization path for LNP models
    lnp_reg_param_id   : smallint  # id
    ---
    smooth_weight      : float     # weight for smoothness regularizer
    sparse_weight      : float     # weight for sparseness regularizer
    """
    contents = [
        [i, a, b] for i, (a, b) in enumerate(
            [[a, b] for a in [0.005, 0.01, 0.02, 0.04, 0.08]
                    for b in [0.02, 0.03, 0.04, 0.05, 0.06]])
    ]



@schema
class LnpFit(dj.Computed):
    definition = """  # LNP model
    -> Region
    -> LnpRegPath
    ---
    log_hash                : varchar(10)   # hash for log directory
    iterations              : int           # numer of iterations
    train_loss              : float         # loss on training set
    val_loss                : float         # loss on validation set
    mse                     : blob          # MSE per cell on test set
    avg_mse                 : float         # average MSE on test set
    corr                    : blob          # correlation between prediction and response
    avg_corr                : float         # average correlation
    var                     : blob          # variance of prediction
    avg_var                 : float         # average variance
    ve                      : blob          # variance explained (1 - MSE/Var)
    avg_ve                  : float         # average VE
    eve                     : blob          # explainable variance explained (1 - MSE/ExplVar)
    avg_eve                 : float         # average EVE
    nnp                     : blob          # normalized noise power (noise var / total var)
    """

    def _make_tuples(self, key):
        lnp = LnpFit.build_net(key)
        learning_rate=0.001
        training = lnp.train(max_iter=10000,
                             val_steps=50,
                             early_stopping_steps=10,
                             learning_rate=learning_rate)
        for (i, (logl, total_loss, pred)) in training:
            print('Step %d | Total loss: %s | Poisson: %s | Var(y): %s' % (i, total_loss, logl, np.mean(np.var(pred, axis=0))))
        print('Done fitting')

        tupl = evaluate_model(lnp, key)
        self.insert1(tupl)


    def get_net(self):
        assert len(self) == 1, 'Relation must be scalar!'
        log_hash, iterations = self.fetch1('log_hash', 'iterations')
        lnp = LnpFit.build_net(list(self.fetch.keys())[0], log_hash=log_hash, global_step=iterations)
        lnp.load_best()
        return lnp


    @staticmethod
    def build_net(key, log_hash=None, global_step=None):
        data = (Region() & key).load_data()
        reg = (LnpRegPath() & key).fetch1()
        lnp = convnet.LNP(data, log_dir='lnp', log_hash=log_hash, global_step=global_step)
        lnp.build(reg['smooth_weight'], reg['sparse_weight'])
        return lnp


    