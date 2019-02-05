from abc import abstractmethod
import time
import random
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def iterate_minibatches(a1,a2,a3):
    return []

import numpy as np

class TrainingContainer:
    def __init__(self, model, opt, batch_size=100, verbose=True):
        self._model = model
        self._opt = opt
        self._batch_size = batch_size
        self._training_loss = []
        self._validation_accuracy = []
        self._verbose = verbose
        self._epoch_loss = []
        self._epoch_acc = []
        self._X_val = None
        self._y_val = None

    @abstractmethod
    def compute_loss(self, X_batch, y_batch):
        pass

    @abstractmethod
    def batch_accuracy(self, X_batch, y_batch):
        pass

    @abstractmethod
    def append_loss(self, loss):
        pass

    def train(self, X_train, y_train, iterable=range(0, 10), X_val=None, y_val=None):
        self._X_val = X_val
        self._y_val = y_val
        if self._X_val is None:
            self._X_val = X_train
            self._y_val = y_train
        for epoch in iterable:
            # In each epoch, we do a full pass over the training data:
            start_time = time.time()
            self._model.train(True)  # enable training mode
            for X_batch, y_batch in iterate_minibatches(X_train, y_train, self._batch_size):
                # train on batch
                loss = self.compute_loss(X_batch, y_batch)
                loss.backward()
                self._opt.step()
                self._opt.zero_grad()
                self.append_loss(loss)

            # And a full pass over the validation data:
            self._model.train(False)  # disable dropout / use averages for batch_norm
            for X_batch, y_batch in iterate_minibatches(self._X_val, self._y_val, self._batch_size):
                self._validation_accuracy.append(self.batch_accuracy(X_batch, y_batch))

            tl = np.mean(self._training_loss[-len(X_train) // self._batch_size:])
            self._epoch_loss.append(tl)
            va = np.mean(self._validation_accuracy[-len(self._X_val) // self._batch_size:]) * 100
            self._epoch_acc.append(va)

            if self._verbose:
                print("Epoch {} took {:.3f}s".format(
                    len(self._epoch_loss), time.time() - start_time))
                print("  training loss (in-iteration): \t{:.6f}".format(tl))
                print("  validation accuracy: \t\t\t{:.2f} %".format(va))


class CudaTrainingContainer(TrainingContainer):
    def compute_loss(self, X_batch, y_batch):
        X_batch = torch.FloatTensor(X_batch).cuda()
        y_batch = torch.LongTensor(y_batch).cuda()
        logits = self._model(X_batch)
        return F.cross_entropy(logits, y_batch).mean()

    def batch_accuracy(self, X_batch, y_batch):
        X_batch = torch.FloatTensor(X_batch).cuda()
        logits = self._model(X_batch).cpu()
        y_pred = logits.argmax(1).data.numpy()
        return np.mean(y_batch == y_pred)

    def append_loss(self, loss):
        self._training_loss.append(loss.cpu().data.numpy())

    def __init__(self, model, opt):
        model.cuda()
        super().__init__(model, opt)


class CPUTrainingContainer(TrainingContainer):

    def compute_loss(self, X_batch, y_batch):
        X_batch = torch.FloatTensor(X_batch)
        y_batch = torch.LongTensor(y_batch)
        logits = self._model(X_batch)
        return F.cross_entropy(logits, y_batch).mean()


    def batch_accuracy(self, X_batch, y_batch):
        X_batch = torch.FloatTensor(X_batch)
        logits = self._model(X_batch)
        y_pred = logits.max(1)[1].data.numpy()
        return np.mean(y_batch == y_pred)

    def append_loss(self, loss):
        self._training_loss.append(loss.data.numpy())

    def __init__(self, model, opt):
        model.cpu()
        super().__init__(model, opt)


def create_k_mask(in_channels, out_channels, kernel, k):
    extended = kernel
    rez = torch.zeros((out_channels, in_channels, *kernel))
    # this is
    o = torch.ones(extended)
    lower = 0
    upper = k
    for i in range(0, out_channels):
        if lower < upper:
            rez[i, lower:upper, :, :] = o
        elif upper == 0:
            rez[i, lower:, :, :] = o
        else:
            rez[i, lower:, :, :] = o
            rez[i, 0:upper, :, :] = o
        lower = (lower + 1) % in_channels
        upper = (upper + 1) % in_channels
    return rez


def create_bernoilli_mask(in_channels, out_channels, kernel, p):
    extended = kernel
    rez = torch.zeros((out_channels, in_channels, *kernel))
    # this is
    o = torch.ones(extended)
    for i in range(0, out_channels):
        rez[i,random.randint(in_channels),:,:] = o
        for j in range(0, in_channels):
            if random.random() < p:
                rez[i,j,:,:] = o
    return rez


class MaskedConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self._mask = mask

    def forward(self, input):
        return F.conv2d(input,
            self._mask * self.weight, # multiply weights by the mask before applying convolution
            self.bias, self.stride, self.padding, self.dilation, self.groups)


class BlockedConv2d(MaskedConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, k, stride=1, padding=0, dilation=1, groups=1, bias=True):
        """
        :param k: number of input channels that affect each output channel
        """
        assert k <= in_channels, "k is bigger than in_channels, the setting is equivalent to a full convolution"
        super().__init__(in_channels, out_channels, kernel_size,
             create_k_mask(in_channels, out_channels, kernel_size, k),
             stride, padding, dilation, groups, bias)


class BernoilliConv2d(MaskedConv2d):

    def __init__(self, in_channels, out_channels, kernel_size, p=0.1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        """
        :param p: the probability that the pair (input_layer, output_layer) is connected
        """
        assert p > 0 and p < 1, "p should be a valid probability. Between 0 and 1"
        super().__init__(in_channels, out_channels, kernel_size,
             create_bernoilli_mask(in_channels, out_channels, kernel_size, p),
             stride, padding, dilation, groups, bias)


factory = lambda: nn.LeakyReLU(negative_slope=0.01)


def construct_net(nonlinearity_factory, conv_factory, n=32):
    """
    creates a Sequential Module that corresponds to the benchmark architecture
    conv3x3(n) -> conv3x3(n) -> max_pool -> conv3x3(1.5 n) -> conv3x3(1.5 n) -> max_pool -> conv3x3(2 n) ->  global_pool -> dense(10, nonlinearity = softmax)
    :param n: well, the n from the formula
    """

    model = nn.Sequential()
    model.add_module('conv_n_0', conv_factory(1, n, kernel_size=(3, 3)))  # 26x26
    model.add_module('nonlinearity_n_0', nonlinearity_factory())
    model.add_module('conv_n_1', conv_factory(n, n, kernel_size=(3, 3)))  # 24x24
    model.add_module('nonlinearity_n_1', nonlinearity_factory())
    #     model.add_module('bn1', nn.BatchNorm2d(n)) # 24x24
    model.add_module('pool_n', nn.MaxPool2d(kernel_size=(2, 2)))  # 12x12

    m = int(1.5 * n)

    model.add_module('conv_1_5n_0', conv_factory(n, m, kernel_size=(3, 3)))  # 10x10
    model.add_module('nonlinearity_1_5n_0', nonlinearity_factory())
    model.add_module('conv_1_5n_1', conv_factory(m, m, kernel_size=(3, 3)))  # 8x8
    model.add_module('nonlinearity_1_5n_1', nonlinearity_factory())
    #     model.add_module('bn2', nn.BatchNorm2d(m)) # 24x24
    model.add_module('pool_1_5n', nn.MaxPool2d(kernel_size=(2, 2)))  # 4x4

    model.add_module('conv_2n_0', conv_factory(m, 2 * n, kernel_size=(3, 3)))  # 2x2
    model.add_module('nonlinearity_2n_0', nonlinearity_factory())
    #     model.add_module('bn3', nn.BatchNorm2d(2*n)) # 24x24
    model.add_module('pool_2n', nn.MaxPool2d(kernel_size=(2, 2)))  # 1x1

    #     model.add_module('global_pool', GlobalMaxPool())
    #     model.add_module('dense0', nn.Linear(1, 10))

    model.add_module('global_pool', Flatten())
    model.add_module('dense0', nn.Linear(2 * n, 10))
    #     model.add_module('dense2_logits', nn.Linear(100, 10)) # logits for 10 classes
    model.apply(xavier_init)

    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3)
    return model, opt


leaky_relu = lambda: nn.LeakyReLU(negative_slope=0.01)
simple_conv = lambda in_channels, out_channels, kernel: nn.Conv2d(in_channels, out_channels, kernel)