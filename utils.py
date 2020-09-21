# -*- coding: utf-8 -*-
"""
/////
"""
import time
import logging
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.module import _addindent
from collections import defaultdict

INT = 0
LONG = 1
FLOAT = 2
EOS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mt_factor', type=float, default=0.1, help='lambda for the second disc')
    parser.add_argument('--irl_net', type=str, default='AIRL_MT', help='AIRL_MT, AIRL_MTSA')
    parser.add_argument('--train_direc', type=str, default='opposite', help='forward, opposite')
    parser.add_argument('--log_dir', type=str, default='log', help='Logging directory')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='model', help='Directory to store model')
    parser.add_argument('--load', type=str, default='', help='File name to load trained model')
    parser.add_argument('--load_user', type=str, default='', help='File name to load user simulator')
    parser.add_argument('--pretrain', type=bool, default=False, help='Set to pretrain')
    parser.add_argument('--test', type=bool, default=False, help='Set to inference')
    parser.add_argument('--config', type=str, default='multiwoz', help='Dataset to use')
    parser.add_argument('--simulator', type=str, default='agenda', help='User simulator to use')
    
    parser.add_argument('--epoch', type=int, default=32, help='Max number of epoch')
    parser.add_argument('--save_per_epoch', type=int, default=1, help="Save model every XXX epoches")
    parser.add_argument('--process', type=int, default=16, help='Process number')
    parser.add_argument('--batchsz', type=int, default=32, help='Batch size')
    parser.add_argument('--batchsz_traj', type=int, default=128, help='Batch size to collect trajectories')
    parser.add_argument('--print_per_batch', type=int, default=400, help="Print log every XXX batches")
    parser.add_argument('--update_round', type=int, default=5, help='Epoch num for inner loop of PPO')
    parser.add_argument('--lr_rl', type=float, default=3e-4, help='Learning rate of dialog policy')
    parser.add_argument('--lr_irl', type=float, default=1e-3, help='Learning rate of reward estimator')
    parser.add_argument('--lr_simu', type=float, default=1e-3, help='Learning rate of user simulator')
    
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted factor')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Clip epsilon of ratio r(theta)')
    parser.add_argument('--tau', type=float, default=0.95, help='Generalized advantage estimation')
    parser.add_argument('--anneal', type=int, default=5000, help='Max steps for annealing')
    parser.add_argument('--clip', type=float, default=0.01, help='Clipping parameter on WGAN')
    parser.add_argument('--s_lambda', type=float, default=0., help='the influence of the state factor')
    parser.add_argument('--temperature', type=float, default=0.01, help='the temperature of soft-argmax')
    parser.add_argument('--action_lambda', type=float, default=0.2, help='the action loss factor')
    parser.add_argument('--gan_type', type=str, default='wgan', help='wgan,vanilla')
    parser.add_argument('--argmax_type', type=str, default='soft', help='soft,gumbel')
    parser.add_argument('--rl_sim', type=str, default='yes', help='yes, no')
    parser.add_argument('--cat_mlp', type=bool, default=False, help='if input the h0 during each decoding step')
    parser.add_argument('--gen_type', type=str, default='beam', help='beam, sample, greedy')
    parser.add_argument('--avg_type', type=str, default='seq', help='seq, real_word, word')
    parser.add_argument('--policy_clip', type=float, default=0.5, help='clip gradient')
    parser.add_argument('--activate_type', type=str, default='tanh', help='tanh, relu, no')

    parser.add_argument('--early_stop', type=bool, default=False, help='early_stop or not')
    parser.add_argument('--patient_increase', type=float, default=3.0, help='patient_increase for early stop')
    parser.add_argument('--improve_threshold', type=float, default=0.996, help='improve_threshold for early stop')
    parser.add_argument('--disc_times', type=int, default=1, help='the freq to train disc')
    parser.add_argument('--data_ratio', type=int, default=100, help='data size ratio')



    return parser 

def init_session(key, cfg):
    turn_data = {}
    turn_data['others'] = {'session_id':key, 'turn':0, 'terminal':False}
    turn_data['sys_action'] = dict()
    turn_data['user_action'] = dict()
    turn_data['history'] = {'sys':dict(), 'user':dict()}
    turn_data['belief_state'] = {'inform':{}, 'request':{}, 'booked':{}}
    for domain in cfg.belief_domains:
        turn_data['belief_state']['inform'][domain] = dict()
        turn_data['belief_state']['request'][domain] = set()
        turn_data['belief_state']['booked'][domain] = ''
    
    session_data = {'inform':{}, 'request':{}, 'book':{}}
    for domain in cfg.belief_domains:
        session_data['inform'][domain] = dict()
        session_data['request'][domain] = set()
        session_data['book'][domain] = False
    
    return turn_data, session_data

def init_goal(dic, goal, cfg):
    for domain in cfg.belief_domains:
        if domain in goal and goal[domain]:
            domain_data = goal[domain]
            # constraint
            if 'info' in domain_data and domain_data['info']:
                for slot, value in domain_data['info'].items():
                    slot = cfg.map_inverse[domain][slot]
                    # single slot value for user goal
                    inform_da = domain+'-'+slot+'-1'
                    if inform_da in cfg.inform_da:
                        dic['inform'][domain][slot] = value
            # booking
            if 'book' in domain_data and domain_data['book']:
                dic['book'][domain] = True
            # request
            if 'reqt' in domain_data and domain_data['reqt']:
                for slot in domain_data['reqt']:
                    slot = cfg.map_inverse[domain][slot]
                    request_da = domain+'-'+slot
                    if request_da in cfg.request_da:
                        dic['request'][domain].add(slot)

def init_logging_handler(log_dir, extra=''):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(log_dir, current_time+extra))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

def to_device(data):
    if type(data) == dict:
        for k, v in data.items():
            data[k] = v.to(device=DEVICE)
    else:
        for idx, item in enumerate(data):
            data[idx] = item.to(device=DEVICE)
    return data

def state_vectorize(state, config, db, noisy=False):
    """
    state: dict_keys(['user_action', 'sys_action', 'user_goal', 'belief_state', 'history', 'others']) 
    state_vec: [user_act, last_sys_act, inform, request, book, degree]
    """
    user_act = np.zeros(len(config.da_usr))
    for da in state['user_action']:
        user_act[config.dau2idx[da]] = 1.
    
    last_sys_act = np.zeros(len(config.da))
    for da in state['last_sys_action']:
        last_sys_act[config.da2idx[da]] = 1.
    
    user_history = np.zeros(len(config.da_usr))
    for da in state['history']['user']:
        user_history[config.dau2idx[da]] = 1.
    
    sys_history = np.zeros(len(config.da))
    for da in state['history']['sys']:
        sys_history[config.da2idx[da]] = 1.
    
    inform = np.zeros(len(config.inform_da))
    for domain in state['belief_state']['inform']:
        for slot, value in state['belief_state']['inform'][domain].items():
            dom_slot, p = domain+'-'+slot+'-', 1
            key = dom_slot + str(p)
            while inform[config.inform2idx[key]]:
                p += 1
                key = dom_slot + str(p)
                if key not in config.inform2idx:
                    break
            else:
                inform[config.inform2idx[key]] = 1.
            
    request = np.zeros(len(config.request_da))
    for domain in state['belief_state']['request']:
        for slot in state['belief_state']['request'][domain]:
            request[config.request2idx[domain+'-'+slot]] = 1.
    
    book = np.zeros(len(config.belief_domains))
    for domain in state['belief_state']['booked']:
        if state['belief_state']['booked'][domain]:
            book[config.domain2idx[domain]] = 1.
    
    degree = db.pointer(state['belief_state']['inform'], config.mapping, config.db_domains, noisy)
        
    final = 1. if state['others']['terminal'] else 0.
    
    state_vec = np.r_[user_act, last_sys_act, user_history, sys_history, inform, request, book, degree, final]
    assert len(state_vec) == config.s_dim
    return state_vec

def action_vectorize(action, config):
    act_vec = np.zeros(config.a_dim)
    for da in action['sys_action']:
        act_vec[config.da2idx[da]] = 1
    # if act_vec.sum()==0:
    #     da = 'general-unk'
    #     act_vec[config.da2idx[da]] = 1
    return act_vec

def action_seq(action, config):
    act_seq = []
    for da in action['sys_action']:
        act_seq.append(config.da2idx[da])
    # if len(act_seq)==0:
    #     da = 'general-unk'
    #     act_seq.append(config.da2idx[da])
    act_seq = sorted(act_seq)
    act_seq.insert(0, 1) # SOS
    act_seq.append(2)  # EOS
    act_seq = pad_to(config.max_len, act_seq, True)
    return act_seq

def count_act(act_seq, config):
    act_cout = defaultdict(int)
    for a_line in act_seq:
        for a in a_line:
            act_name = config.idx2da[int(a.item())]
            act_cout[act_name]+=1
    logging.info(act_cout)
    logging.info(sorted(act_cout, key=act_cout.get, reverse=True))


def pad_to(max_len, tokens, do_pad=True):
    if len(tokens) >= max_len:
        return tokens[0:max_len - 1] + [tokens[-1]]
    elif do_pad:
        return tokens + [0] * (max_len - len(tokens))
    else:
        return tokens

def domain_vectorize(state, config):
    domain_vec = np.zeros(config.domain_dim)
    for da in state['user_action']:
        domain = da.strip().split('-')[0]
        domain_vec[config.domain_index[domain]] = 1
    return domain_vec

def reparameterize(mu, logvar):
    std = (0.5*logvar).exp()
    eps = torch.randn_like(std)
    return eps.mul(std) + mu

def id2onehot(id_list):
    one_hot = []
    sp = id_list.shape
    a_dim = sp[-1]
    if type(id_list)==torch.Tensor:
        id_list = id_list.view(-1).tolist()
    for id in id_list:
        if id==0:
            one_hot += [1,0]
        elif id==1:
            one_hot += [0,1]
        else:
            raise ValueError("id can only be 0 or 1, but got {}".format(id))
    return torch.FloatTensor(one_hot).view(-1, a_dim * 2)

def onehot2id(onehot_list):
    id_list = []
    bs, a_dim = onehot_list.shape
    newlist = onehot_list.view(-1)
    for i in range(0, len(newlist), 2):
        if newlist[i]>= newlist[i+1]:
            id_list.append(0)
        else:
            id_list.append(1)
    return torch.FloatTensor(id_list).view(bs, a_dim//2)

class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack


def summary(model, show_weights=True, show_parameters=True):
    """
    Summarizes torch model by showing trainable parameters and weights.
    """
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params
        # and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = summary(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])
        total_params += params

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ') Total Parameters={}'.format(total_params)
    return tmpstr





class GumbelConnector(nn.Module):
    def __init__(self, use_gpu):
        super(GumbelConnector, self).__init__()
        self.use_gpu = use_gpu

    def sample_gumbel(self, logits, use_gpu, eps=1e-20):
        u = torch.rand(logits.size())
        sample = Variable(-torch.log(-torch.log(u + eps) + eps))
        sample = cast_type(sample, FLOAT, use_gpu)
        return sample

    def gumbel_softmax_sample(self, logits, temperature, use_gpu):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        eps = self.sample_gumbel(logits, use_gpu)
        y = logits + eps
        return F.softmax(y / temperature, dim=y.dim()-1)

    def soft_argmax(self, logits, temperature, use_gpu):
        return F.softmax(logits / temperature, dim=logits.dim()-1)

    def forward(self, logits, temperature=1.0, hard=False,
                return_max_id=False):
        """
        :param logits: [batch_size, n_class] unnormalized log-prob
        :param temperature: non-negative scalar
        :param hard: if True take argmax
        :param return_max_id
        :return: [batch_size, n_class] sample from gumbel softmax
        """
        y = self.soft_argmax(logits, temperature, self.use_gpu)
        # y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        _, y_hard = torch.max(y, dim=1, keepdim=True)
        if hard:
            y_onehot = cast_type(Variable(torch.zeros(y.size())), FLOAT, self.use_gpu)
            y_onehot.scatter_(1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y
    
    def forward_ST(self, logits, temperature=0.8):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.soft_argmax(logits, temperature, self.use_gpu)
        # y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y
    
    def forward_ST_gumbel(self, logits, temperature=0.8):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(torch.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.cuda.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    else:
        if dtype == INT:
            var = var.type(torch.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    return var