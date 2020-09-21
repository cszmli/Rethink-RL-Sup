# -*- coding: utf-8 -*-
"""
/////
"""
import torch
import torch.nn as nn
import numpy as np

from collections import namedtuple
import random
from utils import GumbelConnector, onehot2id


class DiscretePolicy(nn.Module):
    def __init__(self, cfg):
        super(DiscretePolicy, self).__init__()

        self.net = nn.Sequential(nn.Linear(cfg.s_dim, cfg.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.h_dim, cfg.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.h_dim, cfg.a_dim))

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        a_weights = self.net(s)

        return a_weights

    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: [1]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        a_weights = self.forward(s)
        a_probs = torch.softmax(a_weights, 0)

        # randomly sample from normal distribution, whose mean and variance come from policy network.
        # [a_dim] => [1]
        a = a_probs.multinomial(1) if sample else a_probs.argmax(0, True)

        return a

    def get_log_prob(self, s, a):
        """
        :param s: [b, s_dim]
        :param a: [b, 1]
        :return: [b, 1]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.softmax(a_weights, -1)

        # [b, a_dim] => [b, 1]
        trg_a_probs = a_probs.gather(-1, a)
        log_prob = torch.log(trg_a_probs)

        return log_prob
 
    
class MultiDiscretePolicy(nn.Module):
    def __init__(self, cfg, feature_extractor=None):
        super(MultiDiscretePolicy, self).__init__()
        self.cfg = cfg
        self.feature_extractor = feature_extractor
        self.net = nn.Sequential(nn.Linear(cfg.s_dim, cfg.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.h_dim, cfg.a_dim//2))

        self.gumble_length_index = self.gumble_index_multiwoz_binary()
        self.gumble_num = len(self.gumble_length_index)
        self.last_layers = nn.ModuleList()
        for gumbel_width in self.gumble_length_index:
            self.last_layers.append(nn.Linear(cfg.a_dim//2, gumbel_width))
        self.gumbel_connector=GumbelConnector(False)
        
        
    def gumble_index_multiwoz_binary(self):
        # index = 9 * [2]
        index = self.cfg.a_dim * [2]
        return index

    def forward(self, s):
        a_weights = self.net(s)
        input_to_gumbel = []
        for layer, g_width in zip(self.last_layers, self.gumble_length_index):
            out = layer(a_weights)
            out = self.gumbel_connector.forward_ST(out.view(-1,  g_width), self.config.gumbel_temp)
            input_to_gumbel.append(out)
        action_rep = torch.cat(input_to_gumble, -1)
        return action_rep
    
    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: [a_dim]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        action = self.forward(s)
        action = onehot2id(action)
        return action
    
        
    
class MultiDiscDomain(nn.Module):
    def __init__(self, cfg):
        super(MultiDiscDomain, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(cfg.s_dim, cfg.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.h_dim, cfg.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.h_dim, cfg.domain_dim))

    def forward(self, s):
        # [b, s_dim] => [b, domain_dim]
        a_weights = self.net(s)

        return a_weights

class ContinuousPolicy(nn.Module):
    def __init__(self, cfg):
        super(ContinuousPolicy, self).__init__()

        self.net = nn.Sequential(nn.Linear(cfg.s_dim, cfg.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.h_dim, cfg.h_dim),
                                 nn.ReLU())
        self.net_mean = nn.Linear(cfg.h_dim, cfg.a_dim)
        self.net_std = nn.Linear(cfg.h_dim, cfg.a_dim)

    def forward(self, s):
        # [b, s_dim] => [b, h_dim]
        h = self.net(s)

        # [b, h_dim] => [b, a_dim]
        a_mean = self.net_mean(h)
        a_log_std = self.net_std(h)

        return a_mean, a_log_std

    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: [a_dim]
        """
        # forward to get action mean and log_std
        # [s_dim] => [a_dim]
        a_mean, a_log_std = self.forward(s)

        # randomly sample from normal distribution, whose mean and variance come from policy network.
        # [a_dim]
        a = torch.normal(a_mean, a_log_std.exp()) if sample else a_mean

        return a

    def get_log_prob(self, s, a):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :return: [b, 1]
        """
        def normal_log_density(x, mean, log_std):
            """
            x ~ N(mean, std)
            this function will return log(prob(x)) while x belongs to guassian distrition(mean, std)
            :param x:       [b, a_dim]
            :param mean:    [b, a_dim]
            :param log_std: [b, a_dim]
            :return:        [b, 1]
            """
            std = log_std.exp()
            var = std.pow(2)
            log_density = - (x - mean).pow(2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std
        
            return log_density.sum(-1, keepdim=True)
        
        # forward to get action mean and log_std
        # [b, s_dim] => [b, a_dim]
        a_mean, a_log_std = self.forward(s)

        # [b, a_dim] => [b, 1]
        log_prob = normal_log_density(a, a_mean, a_log_std)

        return log_prob
    
    
class Value(nn.Module):
    def __init__(self, cfg):
        super(Value, self).__init__()

        self.net = nn.Sequential(nn.Linear(cfg.s_dim, cfg.hv_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.hv_dim, cfg.hv_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.hv_dim, 1))

    def forward(self, s):
        """
        :param s: [b, s_dim]
        :return:  [b, 1]
        """
        value = self.net(s)

        return value

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward', 'domain'))

class Memory(object):

    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def get_batch(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)
