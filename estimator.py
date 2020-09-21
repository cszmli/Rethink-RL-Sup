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
from utils import to_device, reparameterize
from dbquery import DBQuery

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RewardEstimator(object):
    def __init__(self, args, manager, config, pretrain=False, inference=False, feature_extractor=None):
        self.feature_extractor=feature_extractor
        if args.irl_net=='AIRL_MT':
            self.irl = AIRL_MT(config, args.gamma).to(device=DEVICE)
        elif args.irl_net=='AIRL_MTSA':
            self.irl = AIRL_MTSA(config, args.gamma).to(device=DEVICE)
        else:
            self.irl = AIRL_ORI(config, args.gamma).to(device=DEVICE)
        self.net_type = args.irl_net
        self.train_direc = args.train_direc
        self.mt_factor = args.mt_factor
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()

        self.step = 0
        self.anneal = args.anneal
        self.irl_params = self.irl.parameters()
        self.irl_optim = optim.RMSprop(self.irl_params, lr=args.lr_irl)
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
    
    def irl_loop(self, data_real, data_gen):
        s_real, a_real, next_s_real, d_real = to_device(data_real)
        s, a, next_s, d_fake = data_gen

        # train with real data
        if self.net_type!='AIRL_ORI':
            s_real = self.feature_extractor.feature_extractor(s_real).detach()
            next_s_real = self.feature_extractor.feature_extractor(next_s_real).detach()
        weight_real, weight_domain_real = self.irl(s_real, a_real, next_s_real)
        loss_real = -weight_real.mean()

        # train with generated data
        if self.net_type!='AIRL_ORI':
            s = self.feature_extractor.feature_extractor(s).detach()
            next_s = self.feature_extractor.feature_extractor(next_s).detach()
        weight, weight_domain_fake = self.irl(s, a, next_s)
        loss_gen = weight.mean()
        
        loss_d = torch.tensor([0.])

        return loss_real, loss_gen, loss_d

    
    def train_irl(self, batch, epoch):
        self.irl.train()
        input_s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        input_a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        input_next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        input_d = torch.from_numpy(np.stack(batch.domain)).to(device=DEVICE)

        batchsz = input_s.size(0)
        
        real_loss, gen_loss, domain_loss = 0., 0., 0.
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
            
            self.irl_optim.zero_grad()
            loss_real, loss_gen, loss_domain = self.irl_loop(data, (s, a, next_s, d))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            domain_loss += loss_domain.item()
            loss = loss_real + loss_gen + loss_domain * self.mt_factor
            loss.backward()
            self.irl_optim.step()
            
            for p in self.irl_params:
                p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
            
        real_loss /= turns
        gen_loss /= turns
        domain_loss /= turns
        logging.debug('<<reward estimator>> epoch {}, loss_real:{}, loss_gen:{}, loss_domain:{}'.format(
                epoch, real_loss, gen_loss, domain_loss))
        if (epoch+1) % self.save_per_epoch == 0:
            self.save_irl(self.save_dir, epoch)
        self.irl.eval()
    
    def test_irl(self, batch, epoch, best):
        input_s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        input_a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        input_next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        input_d = torch.from_numpy(np.stack(batch.domain)).to(device=DEVICE)        
        
        batchsz = input_s.size(0)
        
        real_loss, gen_loss, domain_loss = 0., 0., 0.
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
            
            loss_real, loss_gen, loss_domain = self.irl_loop(data, (s, a, next_s, d))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            domain_loss += loss_domain.item()
            
        real_loss /= turns
        gen_loss /= turns
        domain_loss /= turns
        logging.debug('<<reward estimator>> validation, epoch {}, loss_real:{}, loss_gen:{}, loss_domain:{}'.format(
                epoch, real_loss, gen_loss, domain_loss))
        loss = real_loss + gen_loss + self.mt_factor * domain_loss
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
            
            loss_real, loss_gen, loss_domain = self.irl_loop(data, (s, a, next_s, d))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            domain_loss += loss_domain.item()
            
        real_loss /= turns
        gen_loss /= turns
        domain_loss /= turns
        logging.debug('<<reward estimator>> test, epoch {}, loss_real:{}, loss_gen:{}, loss_domain:{}'.format(
                epoch, real_loss, gen_loss, domain_loss))
        return best
    
    def update_irl(self, inputs, batchsz, epoch, best=None):
        """
        train the reward estimator (together with encoder) using cross entropy loss (real, mixed, generated)
        Args:
            inputs: (s, a, next_s)
        """
        backward = True if best is None else False
        if backward:
            self.irl.train()
        input_s, input_a, input_next_s, input_d = inputs
        
        real_loss, gen_loss, domain_loss = 0., 0., 0.
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
            loss_real, loss_gen, loss_domain = self.irl_loop(data, (s, a, next_s, d))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            domain_loss += loss_domain.item()
            if backward:
                loss =loss_real + loss_gen + loss_domain * self.mt_factor
                loss.backward()
                self.irl_optim.step()
                
                for p in self.irl_params:
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
        
        real_loss /= turns
        gen_loss /= turns
        domain_loss /= turns
        if backward:
            logging.debug('<<reward estimator>> epoch {}, loss_real:{}, loss_gen:{}, loss_domain:{}'.format(
                    epoch, real_loss, gen_loss, domain_loss))
            self.irl.eval()
        else:
            logging.debug('<<reward estimator>> validation, epoch {}, loss_real:{}, loss_gen:{}, loss_domain:{}'.format(
                    epoch, real_loss, gen_loss, domain_loss))
            loss = real_loss + gen_loss + self.mt_factor * domain_loss
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
    
    def estimate(self, s, a, next_s, log_pi):
        """
        infer the reward of state action pair with the estimator
        """
        if self.net_type!='AIRL_ORI':
            s = self.feature_extractor.feature_extractor(s).detach()
            next_s = self.feature_extractor.feature_extractor(next_s).detach()
        weight, _ = self.irl(s, a.float(), next_s)
        logging.debug('<<reward estimator>> weight {}'.format(weight.mean().item()))
        logging.debug('<<reward estimator>> log pi {}'.format(log_pi.mean().item()))
        # see AIRL paper
        # r = f(s, a, s') - log_p(a|s)
        reward = (weight - log_pi).squeeze(-1)
        return reward

class AIRL(nn.Module):
    """
    label: 1 for real, 0 for generated
    """
    def __init__(self, cfg, gamma):
        super(AIRL, self).__init__()
        
        self.gamma = gamma
        self.g = nn.Sequential(nn.Linear(cfg.s_dim+cfg.a_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
        self.h = nn.Sequential(nn.Linear(cfg.s_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
    
    def forward(self, s, a, next_s):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :param next_s: [b, s_dim]
        :return:  [b, 1]
        """
        weights = self.g(torch.cat([s,a], -1)) + self.gamma * self.h(next_s) - self.h(s)
        return weights



class AIRL_MT(nn.Module):
    """
    label: 1 for real, 0 for generated
    """
    def __init__(self, cfg, gamma):
        super(AIRL_MT, self).__init__()
        
        self.gamma = gamma
        self.g = nn.Sequential(
                               nn.Linear(cfg.h_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
        self.h = nn.Sequential(
                               nn.Linear(cfg.h_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
    def forward(self, s, a, next_s):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :param next_s: [b, s_dim]
        :return:  [b, 1]
        """
        hidden = s
        hidden_next = next_s
        weights = self.g(hidden) + self.gamma * self.h(hidden_next) - self.h(hidden)
        domain_weights = 0.
        return weights, domain_weights
    

class AIRL_MTSA(nn.Module):
    """
    label: 1 for real, 0 for generated
    """
    def __init__(self, cfg, gamma):
        super(AIRL_MTSA, self).__init__()
        
        self.gamma = gamma                            
        self.a = nn.Sequential(nn.Linear(cfg.a_dim, cfg.hi_dim),
                               nn.ReLU())
        self.g = nn.Sequential(
                               nn.Linear(cfg.h_dim + cfg.hi_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
        self.h = nn.Sequential(
                               nn.Linear(cfg.h_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
    def forward(self, s, a, next_s):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :param next_s: [b, s_dim]
        :return:  [b, 1]
        """
        hidden = s
        hidden_next = next_s
        hidden_a = self.a(a)
        hidden_sa = torch.cat([hidden, hidden_a], -1)
        weights = self.g(hidden_sa) + self.gamma * self.h(hidden_next) - self.h(hidden)
        domain_weights = 0.
        return weights, domain_weights


class AIRL_ORI(nn.Module):
    """
    label: 1 for real, 0 for generated
    """
    def __init__(self, cfg, gamma):
        super(AIRL_ORI, self).__init__()
        
        self.gamma = gamma

        self.g = nn.Sequential(nn.Linear(cfg.s_dim + cfg.a_dim, cfg.hi_dim),
                               nn.ReLU(),
                            #    nn.Linear(cfg.hi_dim, cfg.hi_dim),
                            #    nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
        self.h = nn.Sequential(nn.Linear(cfg.s_dim, cfg.hi_dim),
                               nn.ReLU(),
                            #    nn.Linear(cfg.hi_dim, cfg.hi_dim),
                            #    nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))

    
    def forward(self, s, a, next_s):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :param next_s: [b, s_dim]
        :return:  [b, 1]
        """
        sa = torch.cat([s, a], -1)
        weights = self.g(sa) + self.gamma * self.h(next_s) - self.h(s)
        domain_weights = 0.
        return weights, domain_weights



class DISC_MTSA(nn.Module):
    """
    label: 1 for real, 0 for generated
    """
    def __init__(self, cfg, gamma):
        super(DISC_MTSA, self).__init__()
        
        self.gamma = gamma
        self.c = nn.Sequential(nn.Linear(cfg.s_dim, cfg.hi_dim),
                               nn.ReLU())
        self.a = nn.Sequential(nn.Linear(cfg.a_dim, cfg.hi_dim),
                               nn.ReLU())
        self.g = nn.Sequential(
                               nn.Linear(cfg.hi_dim + cfg.hi_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
        self.h = nn.Sequential(
                               nn.Linear(cfg.hi_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
        self.disc = nn.Sequential(
                               nn.Linear(cfg.hi_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, cfg.domain_dim))
    
    def forward(self, s, a, next_s):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :param next_s: [b, s_dim]
        :return:  [b, 1]
        """
        hidden = self.c(s)
        hidden_next = self.c(next_s)
        hidden_a = self.a(a)
        hidden_sa = torch.cat([hidden, hidden_a], -1)
        weights = self.g(hidden_sa) + self.gamma * self.h(hidden_next) - self.h(hidden)
        domain_weights = self.disc(hidden)
        return weights, domain_weights