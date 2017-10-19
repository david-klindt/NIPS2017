import numpy as np
import os
import inspect


class Dataset:
    def __init__(self, images, responses, images_test, responses_test, seed=None, train_frac=0.8):
        # normalize images
        imgs_mean = np.mean(images, axis=0)
        imgs_sd = np.std(images, axis=0)
        px_x = int(np.sqrt(images.shape[1]))
        normalize = lambda imgs: ((imgs - imgs_mean) / imgs_sd).reshape([-1, px_x, px_x])
        self.images = normalize(images)[...,None]
        self.images_test = normalize(images_test)[...,None]
        self.responses = responses
        self.responses_test = responses_test
        self.num_neurons = responses.shape[1]
        self.num_images = responses.shape[0]
        self.px_x = self.images.shape[2]
        self.px_y = self.images.shape[1]
        if seed:
            np.random.seed(seed)
        perm = np.random.permutation(self.num_images)
        self.train_idx = sorted(perm[:round(self.num_images * train_frac)])
        self.val_idx = sorted(perm[round(self.num_images * train_frac):])
        self.num_train_samples = len(self.train_idx)
        self.next_epoch()

    def val(self):
        return self.images[self.val_idx], self.responses[self.val_idx]

    def train(self):
        return self.images[self.train_idx], self.responses[self.train_idx]

    def test(self, averages=True):
        return self.images_test, self.responses_test.mean(axis=0) if averages else self.responses_test

    def minibatch(self, batch_size):
        im = self.images[self.train_idx]
        res = self.responses[self.train_idx]
        if self.minibatch_idx + batch_size > len(self.train_perm):
            self.next_epoch()
        idx = self.train_perm[self.minibatch_idx + np.arange(0, batch_size)]
        self.minibatch_idx += batch_size
        return im[idx, :, :], res[idx, :]

    def next_epoch(self):
        self.minibatch_idx = 0
        self.train_perm = np.random.permutation(self.num_train_samples)


def load_data(region_num):
    ''' Load data from disk.
    
    Parameters:
        region_num: Region number in dataset (1, 2 or 3)
    
    Outpus:
        data: Dataset object (see database/Dataset)
    '''
    path = os.path.dirname(inspect.stack()[0][1])
    path = os.path.join(path, 'Data/region%d' % region_num)
    imgs = np.load(os.path.join(path, 'training_inputs.npy'))
    imgs_test = np.load(os.path.join(path, 'validation_inputs.npy'))    # odd naming convention (val and test backwards)
    responses = np.load(os.path.join(path, 'training_set.npy'))
    responses_test = np.load(os.path.join(path, 'raw_validation_set.npy'))
    return Dataset(imgs, responses, imgs_test, responses_test, seed=1)

