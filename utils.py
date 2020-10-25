import os
import sys
import shutil
import scipy.stats as stats
import numpy as np
import torch.nn as nn

def printer(status, epoch, num_epochs, batch, num_batchs, loss, loss_mean, acc, acc_mean):
    sys.stdout.write("\r[{}]-[Epoch {}/{}] [Batch {}/{}] [Loss: {:.2f} (mean: {:.2f}), Acc: {:.2f}% (mean: {:.2f}%)]".format(
            status,
            epoch,
            num_epochs,
            batch,
            num_batchs,
            loss,
            loss_mean,
            acc,
            acc_mean
        )
    )

def args_print(args):
    # print
    print("=================================================")
    [print("{}:{}".format(arg, getattr(args, arg))) for arg in vars(args)]
    print("=================================================")

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h # m +-h

def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False

def freeze_layer(model, layer):
    n = 0
    for module in model.children():
        n += 1
        if n == num:
            freeze_all(module)

def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm2d) :
            module.eval()

def initialize_linear(model):
    if type(model) == nn.Linear:
        nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)