import numpy as np


# for pruning
def acc_reward(net, acc, flops):
    return acc * 0.01

def acc_reward_mxd(net, acc, flops):
    return (acc * 0.01 - 0.85) * 10


def acc_flops_reward(net, acc, flops):
    error = (100 - acc) * 0.01
    return -error * np.log(flops)

def conditional1(net,beta, acc, flops):
    if beta == .3:
        return acc* 0.01-0.614
    if beta == .5:
        return acc* 0.01-0.85
    if beta == .7:
        return acc* 0.01-0.90
def conditional2(net,beta, acc, flops):
    if beta == .3:
        return acc* 0.01/0.614
    if beta == .5:
        return acc* 0.01/0.85
    if beta == .7:
        return acc* 0.01/0.90