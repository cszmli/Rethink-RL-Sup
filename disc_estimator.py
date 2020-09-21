# -*- coding: utf-8 -*-
"""
/////
"""
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils import to_device, reparameterize, id2onehot, summary
from dbquery import DBQuery
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiscEstimator(object):
    def __init__(self, args, manager, config, pretrain=False, inference=False):
        self.disc_type = args.gan_type
        if self.disc_type=='vanilla':
            self.irl = DISC_Vanilla(config, args.gamma).to(device=DEVICE)
        elif self.disc_type=='wgan':
            self.irl = DISC_MTSA(config, args.gamma).to(device=DEVICE)
        else:
            raise ValueError(" No such type: {}".format(self.disc_type))
        logging.info(summary(self.irl, show_weights=False))
        self.net_type = args.irl_net
        self.train_direc = args.train_direc
        self.mt_factor = args.mt_factor
        self.action_lambda = 0
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()
        self.loss_BCE = nn.BCELoss()

        self.step = 0
        self.anneal = args.anneal
        self.irl_params = self.irl.parameters()
        self.irl_optim = optim.Adam(self.irl_params, lr=args.lr_irl)
        self.weight_cliping_limit = args.clip
        
        self.save_dir = args.save_dir
        self.save_per_epoch = args.save_per_epoch
        self.optim_batchsz = args.batchsz
        self.irl.eval()
        
        db = DBQuery(args.data_dir)
        if pretrain:
            self.print_per_batch = args.print_per_batch
            self.data_train = manager.create_dataset_irl('train', args.batchsz, config, db)
            self.data_valid = manager.create_dataset_irl('valid', args.batchsz, config, db)
            self.data_test = manager.create_dataset_irl('test', args.batchsz, config, db)
            self.irl_iter = iter(self.data_train)
            self.irl_iter_valid = iter(self.data_valid)
            self.irl_iter_test = iter(self.data_test)
        elif not inference:
            self.data_train = manager.create_dataset_irl('train', args.batchsz, config, db)
            self.data_valid = manager.create_dataset_irl('valid', args.batchsz, config, db)
            self.irl_iter = iter(self.data_train)
            self.irl_iter_valid = iter(self.data_valid)
        
    def kl_divergence(self, mu, logvar, istrain):
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        beta = min(self.step/self.anneal, 1) if istrain else 1
        return beta*klds
    
    def disc_loop(self, data_real, data_gen, shuffle=False):
        flip = False
        s_real, a_real_, next_s_real, d_real = to_device(data_real)
        s, a, next_s, d_fake = data_gen
        a_real = id2onehot(a_real_)

        if np.random.random()<0.3:
            flip = True
            s, a, d_fake = s_real, a_real_[:,torch.randperm(a_real_.size()[1])], d_real
        a = id2onehot(a)
        # train with real data
        weight_real, s_trust_real = self.irl(s_real.detach(), a_real.detach())
        loss_real = self.loss_BCE(weight_real.view(-1), torch.ones_like(weight_real.view(-1)).detach()).mean()
        # train with generated data
        weight, s_trust_fake = self.irl(s, a)
        loss_gen = self.loss_BCE(weight.view(-1), torch.zeros_like(weight.view(-1)).detach()).mean()

        s_loss_real = self.loss_BCE(s_trust_real.view(-1), torch.ones_like(s_trust_real.view(-1)).detach()).mean()
        s_loss_gen = self.loss_BCE(s_trust_fake.view(-1), torch.zeros_like(s_trust_fake.view(-1)).detach()).mean()
        
        s_loss = s_loss_gen + s_loss_real

        return loss_real, loss_gen, s_loss  * self.action_lambda


    def irl_loop(self, data_real, data_gen, shuffle=False):
        flip = False
        s_real, a_real_, next_s_real, d_real = to_device(data_real)
        s, a, next_s, d_fake = data_gen
        a_real = id2onehot(a_real_)

        if np.random.random()<0.3:
            flip = True
            s, a, d_fake = s_real, a_real_[:,torch.randperm(a_real_.size()[1])], d_real
        a = id2onehot(a)
        # train with real data
        weight_real, s_trust_real = self.irl(s_real.detach(), a_real.detach())
        # loss_real = self.loss_BCE(weight_real.view(-1), torch.ones_like(weight_real.view(-1)).detach()).mean()
        loss_real = -weight_real.mean()
        # train with generated data
        weight, s_trust_fake = self.irl(s, a)
        # loss_gen = self.loss_BCE(weight.view(-1), torch.zeros_like(weight.view(-1)).detach()).mean()
        loss_gen = weight.mean()
        
        s_loss = -s_trust_real.mean() + s_trust_fake.mean()
        # s_loss_real = self.loss_BCE(s_trust_real.view(-1), torch.ones_like(s_trust_real.view(-1)).detach()).mean()
        # s_loss_gen = self.loss_BCE(s_trust_fake.view(-1), torch.zeros_like(s_trust_fake.view(-1)).detach()).mean()
        # s_loss = s_loss_gen + s_loss_real

        return loss_real, loss_gen, s_loss * self.action_lambda

    
    def train_disc(self, batch, epoch):
        self.irl.train()
        input_s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        input_a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        input_next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        input_d = torch.from_numpy(np.stack(batch.domain)).to(device=DEVICE)

        batchsz = input_s.size(0)
        
        real_loss, gen_loss, state_loss = 0., 0., 0.
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
        next_s_chunk = torch.chunk(input_next_s, turns)
        d_chunk = torch.chunk(input_d, turns)
        
        for s, a, next_s, d in zip(s_chunk, a_chunk, next_s_chunk, d_chunk):
            try:
                data = self.irl_iter.next()
            except StopIteration:
                self.irl_iter = iter(self.data_train)
                data = self.irl_iter.next()
            
            # for p in self.irl_params:
            #     p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)


            self.irl_optim.zero_grad()
            if self.disc_type=='vanilla':
                loss_real, loss_gen, loss_s = self.disc_loop(data, (s, a, next_s, d))
            elif self.disc_type=='wgan':
                loss_real, loss_gen, loss_s = self.irl_loop(data, (s, a, next_s, d))
            else:
                raise ValueError("No such disc type: {}".format(self.disc_type))

            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            state_loss += loss_s.item()
            loss = loss_real + loss_gen + loss_s
            loss.backward()

            if self.disc_type=='vanilla':
                torch.nn.utils.clip_grad_norm_(self.irl_params, 10)

            self.irl_optim.step()
            
            if self.disc_type=='wgan':
                for p in self.irl_params:
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
            
        real_loss /= turns
        gen_loss /= turns
        state_loss /= turns
        logging.debug('<<reward estimator>> epoch {}, loss_real:{}, loss_gen:{}, loss_state:{}'.format(
                epoch, real_loss, gen_loss, state_loss))
        if (epoch+1) % self.save_per_epoch == 0:
            self.save_irl(self.save_dir, epoch)
        self.irl.eval()
    
    def test_disc(self, batch, epoch, best):
        input_s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        input_a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        input_next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        input_d = torch.from_numpy(np.stack(batch.domain)).to(device=DEVICE)        
        
        batchsz = input_s.size(0)
        
        real_loss, gen_loss, state_loss = 0., 0., 0.
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
        next_s_chunk = torch.chunk(input_next_s, turns)
        d_chunk = torch.chunk(input_d, turns)
        
        for s, a, next_s, d in zip(s_chunk, a_chunk, next_s_chunk, d_chunk):
            try:
                data = self.irl_iter_valid.next()
            except StopIteration:
                self.irl_iter_valid = iter(self.data_valid)
                data = self.irl_iter_valid.next()
            
            if self.disc_type=='vanilla':
                loss_real, loss_gen, loss_s = self.disc_loop(data, (s, a, next_s, d))
            elif self.disc_type=='wgan':
                loss_real, loss_gen, loss_s = self.irl_loop(data, (s, a, next_s, d))
            else:
                raise ValueError("No such disc type: {}".format(self.disc_type))

            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            state_loss += loss_s.item()
            
        real_loss /= turns
        gen_loss /= turns
        state_loss /= turns
        logging.debug('<<reward estimator>> validation, epoch {}, loss_real:{}, loss_gen:{}, loss_state:{}'.format(
                epoch, real_loss, gen_loss, state_loss))
        loss = real_loss + gen_loss
        if loss < best:
            logging.info('<<reward estimator>> best model saved')
            best = loss
            self.save_irl(self.save_dir, 'best')
            
        for s, a, next_s, d in zip(s_chunk, a_chunk, next_s_chunk, d_chunk):
            try:
                data = self.irl_iter_test.next()
            except StopIteration:
                self.irl_iter_test = iter(self.data_test)
                data = self.irl_iter_test.next()
            
            if self.disc_type=='vanilla':
                loss_real, loss_gen, loss_s = self.disc_loop(data, (s, a, next_s, d))
            elif self.disc_type=='wgan':
                loss_real, loss_gen, loss_s = self.irl_loop(data, (s, a, next_s, d))
            else:
                raise ValueError("No such disc type: {}".format(self.disc_type))

            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            state_loss += loss_s.item()
            
        real_loss /= turns
        gen_loss /= turns
        state_loss /= turns
        logging.debug('<<reward estimator>> test, epoch {}, loss_real:{}, loss_gen:{}, loss_state:{}'.format(
                epoch, real_loss, gen_loss, state_loss))
        return best
    
    def update_disc(self, inputs, batchsz, epoch, best=None):
        """
        train the reward estimator (together with encoder) using cross entropy loss (real, mixed, generated)
        Args:
            inputs: (s, a, next_s)
        """
        backward = True if best is None else False
        if backward:
            self.irl.train()
        input_s, input_a, input_next_s, input_d = inputs
        
        real_loss, gen_loss, state_loss = 0., 0., 0.
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
        next_s_chunk = torch.chunk(input_next_s, turns)
        d_chunk = torch.chunk(input_d, turns)

        
        for s, a, next_s, d in zip(s_chunk, a_chunk, next_s_chunk, d_chunk):
            if backward:
                try:
                    data = self.irl_iter.next()
                except StopIteration:
                    self.irl_iter = iter(self.data_train)
                    data = self.irl_iter.next()
            else:
                try:
                    data = self.irl_iter_valid.next()
                except StopIteration:
                    self.irl_iter_valid = iter(self.data_valid)
                    data = self.irl_iter_valid.next()
            

            if backward:
                self.irl_optim.zero_grad()

            if self.disc_type=='vanilla':
                loss_real, loss_gen, loss_s = self.disc_loop(data, (s, a, next_s, d))
            elif self.disc_type=='wgan':
                loss_real, loss_gen, loss_s = self.irl_loop(data, (s, a, next_s, d))
            else:
                raise ValueError("No such disc type: {}".format(self.disc_type))

            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            state_loss += loss_s.item()
            if backward:
                loss =loss_real + loss_gen + loss_s
                loss.backward()
                self.irl_optim.step()

                if self.disc_type=='vanilla':
                    torch.nn.utils.clip_grad_norm_(self.irl_params, 10)

                self.irl_optim.step()
            
                if self.disc_type=='wgan':
                    for p in self.irl_params:
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

        real_loss /= turns
        gen_loss /= turns
        state_loss /= turns
        if backward:
            logging.debug('<<reward estimator>> epoch {}, loss_real:{}, loss_gen:{}, loss_state:{}'.format(
                    epoch, real_loss, gen_loss, state_loss))
            self.irl.eval()
        else:
            logging.debug('<<reward estimator>> validation, epoch {}, loss_real:{}, loss_gen:{}, loss_state:{}'.format(
                    epoch, real_loss, gen_loss, state_loss))
            loss = real_loss + gen_loss  
            if loss < best:
                logging.info('<<reward estimator>> best model saved')
                best = loss
                self.save_irl(self.save_dir, 'best')
            return best
        
    def save_irl(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.irl.state_dict(), directory + '/' + str(epoch) + '_estimator.mdl')
        logging.info('<<reward estimator>> epoch {}: saved network to mdl'.format(epoch))
        
    def load_irl(self, filename):
        irl_mdl = filename + '_estimator.mdl'
        if os.path.exists(irl_mdl):
            self.irl.load_state_dict(torch.load(irl_mdl))
            logging.info('<<reward estimator>> loaded checkpoint from file: {}'.format(irl_mdl))
    
    def estimate(self, s, a):
        """
        infer the reward of state action pair with the estimator
        """
        weight, _ = self.irl(s, a)
        return weight

    


class DISC_MTSA(nn.Module):
    """
    label: 1 for real, 0 for generated
    """
    def __init__(self, cfg, gamma):
        super(DISC_MTSA, self).__init__()
        self.s_lambda = cfg.s_lambda
        sa_dim = cfg.s_dim+ cfg.a_dim * 2
        # logging.info("disc input: {}".format(sa_dim))
        # sa_dim = cfg.hi_dim + cfg.hi_dim
        self.gamma = gamma
        self.c = nn.Sequential(nn.Linear(cfg.s_dim, cfg.hi_dim),
                               nn.ReLU())
        self.a = nn.Sequential(nn.Linear(cfg.a_dim * 2, cfg.hi_dim),
                               nn.ReLU())
        self.g = nn.Sequential(
                               nn.Linear(sa_dim, sa_dim//2),
                               nn.ReLU(),
                               nn.Linear(sa_dim//2, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
        self.disc_a = nn.Sequential(
                               nn.Linear(cfg.a_dim * 2, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
        self.disc_s = nn.Sequential(
                               nn.Linear(cfg.s_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
        self.disc = nn.Sequential(
                               nn.Linear(cfg.s_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, cfg.domain_dim))
    
    def forward(self, s, a):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :param next_s: [b, s_dim]
        :return:  [b, 1]
        """
        # hidden = self.c(s)
        # hidden_a = self.a(a)
        # hidden_sa = torch.cat([hidden, hidden_a], -1)
        hidden_sa = torch.cat([s, a], -1)

        # hidden_sa = torch.cat([hidden, a], -1)
        # weights = torch.sigmoid(self.g(hidden_sa))
        weights = self.g(hidden_sa)
        a_value = self.disc_a(a)
        s_value = torch.sigmoid(self.disc_s(s))
        s_factor = s_value.detach() + (1.0 - s_value.detach()) * (1.0 - self.s_lambda)
        return weights, a_value
        # domain_weights = self.disc(s)
        # return weights * s_factor + a_value, s_value

class DISC_Vanilla(nn.Module):
    """
    label: 1 for real, 0 for generated
    """
    def __init__(self, cfg, gamma):
        super(DISC_Vanilla, self).__init__()
        self.s_lambda = cfg.s_lambda
        sa_dim = cfg.s_dim+ cfg.a_dim * 2
        # sa_dim = cfg.hi_dim + cfg.hi_dim
        self.gamma = gamma
        self.c = nn.Sequential(nn.Linear(cfg.s_dim, cfg.hi_dim),
                               nn.ReLU())
        self.a = nn.Sequential(nn.Linear(cfg.a_dim * 2, cfg.hi_dim),
                               nn.ReLU())
        self.g = nn.Sequential(
                               nn.Linear(sa_dim, sa_dim//2),
                               nn.ReLU(),
                               nn.Linear( sa_dim//2,  sa_dim//4),
                               nn.ReLU(),
                               nn.Linear(sa_dim//4, 1))
        self.disc_a = nn.Sequential(
                               nn.Linear(cfg.a_dim * 2, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
        self.disc_s = nn.Sequential(
                               nn.Linear(cfg.s_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
        self.disc = nn.Sequential(
                               nn.Linear(cfg.s_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, cfg.domain_dim))
    
    def forward(self, s, a):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :param next_s: [b, s_dim]
        :return:  [b, 1]
        """
        # hidden = self.c(s)
        # hidden_a = self.a(a)
        # hidden_sa = torch.cat([hidden, hidden_a], -1)
        hidden_sa = torch.cat([s, a], -1)
        weights = torch.sigmoid(self.g(hidden_sa))
        a_value = torch.sigmoid(self.disc_a(a)) 
        s_value = torch.sigmoid(self.disc_s(s))
        s_factor = s_value.detach() + (1.0 - s_value.detach()) * (1.0 - self.s_lambda)
        return weights, a_value
