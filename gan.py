# -*- coding: utf-8 -*-
"""
/////
"""
import os
import numpy as np
import logging
import pickle
import copy
import torch
import random
import torch.nn as nn
from torch import optim
from torch import multiprocessing as mp
from utils import GumbelConnector, onehot2id, id2onehot
from rlmodule import Value, Memory, Transition
from estimator import RewardEstimator
from disc_estimator import DiscEstimator
from utils import state_vectorize, to_device, domain_vectorize, summary
from metrics import Evaluator
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class human_reward(object):
    def __init__(self, args, config):
        self.evaluator = Evaluator(args.data_dir, config)
    
    def check_success(self, s):
        match_session = self.evaluator.match_rate(s, True)
        inform_session = self.evaluator.inform_F1(s, True)
        if (match_session == 1 and inform_session[1] == 1) \
        or (match_session == 1 and inform_session[1] is None) \
        or (match_session is None and inform_session[1] == 1):
            return True
        else:
            return False

    def reward_human(self, s, done):
        success = self.check_success(s)
        if success:
            reward = 0.2 * 40 
        if not success and not done:
            reward = -0.1
        if not success and done:
            reward = -0.1 * 40  # 10
        return reward

def sampler(pid, queue, evt, env, policy, batchsz, human_reward):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()
    # human_reward = human_reward()

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 40
    real_traj_len = 0

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()

        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(state_vectorize(s, env.cfg, env.db))
            a = policy.select_action(s_vec.to(device=DEVICE)).cpu()
            # print(a.shape)
            d = torch.Tensor(domain_vectorize(s, env.cfg))
            # interact with env
            next_s, done = env.step(s, a)

            # a flag indicates ending or not
            mask = 0 if done else 1
            
            # get reward compared to demostrations
            next_s_vec = torch.Tensor(state_vectorize(next_s, env.cfg, env.db))
            r = human_reward.reward_human(next_s, done)
            
            # save to queue
            buff.push(s_vec.numpy(), a.numpy(), mask, next_s_vec.numpy(), r, d.numpy())

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()


class GAN(object):
    def __init__(self, env_cls, args, manager, cfg, process_num, pre=False, pre_irl=False, infer=False, realtrain=False):
        """
        :param env_cls: env class or function, not instance, as we need to create several instance in class.
        :param args:
        :param manager:
        :param cfg:
        :param process_num: process number
        :param pre: set to pretrain mode
        :param infer: set to test mode
        """
        self.cfg= cfg
        self.train_direc = args.train_direc
        self.realtrain = realtrain
        self.process_num = process_num
        self.human_reward = human_reward(args, cfg)
        self.gan_type = args.gan_type
        self.disc_times = args.disc_times
        # initialize envs for each process
        self.env_list = []
        for _ in range(process_num):
            self.env_list.append(env_cls())

        self.policy = MultiDiscretePolicy(cfg).to(device=DEVICE)
        self.value = Value(cfg).to(device=DEVICE)
        logging.info(summary(self.policy, show_weights=False))
        # logging.info(summary(self.value , show_weights=False))
        
        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()
        self.loss = nn.BCELoss()
        
        if pre:
            self.print_per_batch = args.print_per_batch
            from dbquery import DBQuery
            db = DBQuery(args.data_dir)
            self.data_train = manager.create_dataset_rl('train', args.batchsz, cfg, db)
            self.data_valid = manager.create_dataset_rl('valid', args.batchsz, cfg, db)
            self.data_test = manager.create_dataset_rl('test', args.batchsz, cfg, db)
            # self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()
        else:
            # self.rewarder = RewardEstimator(args, manager, cfg, pretrain=pre_irl, inference=infer, feature_extractor=self.policy)
            self.rewarder = DiscEstimator(args, manager, cfg, pretrain=pre_irl, inference=infer)
            self.evaluator = Evaluator(args.data_dir, cfg)
            from dbquery import DBQuery
            db = DBQuery(args.data_dir)
            self.data_train = manager.create_dataset_rl('train', args.batchsz, cfg, db)
        self.expert_iter = iter(self.data_train)
        
        self.save_dir = args.save_dir
        self.save_per_epoch = args.save_per_epoch
        self.optim_batchsz = args.batchsz
        self.update_round = args.update_round
        self.policy.eval()
        
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.tau = args.tau
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=args.lr_rl)
        self.value_optim = optim.Adam(self.value.parameters(), lr=args.lr_rl)
        self.mt_factor = args.mt_factor
    
    def retrieve_expert(self, to_device_=True):
        try:
            data = self.expert_iter.next()
        except StopIteration:
            self.expert_iter = iter(self.data_train)
            data = self.expert_iter.next()
        if to_device_:
            return to_device(data)
        else:
            return data
            

    def policy_loop(self, data):
        s, target_a, target_d = to_device(data)
        target_a = id2onehot(target_a)
        a_weights = self.policy(s)
        loss_a = self.multi_entropy_loss(a_weights, target_a)
        return loss_a
    
    def imitate_loop(self, data):
        s, target_a, target_d = to_device(data)
        target_a = id2onehot(target_a)
        a_weights = self.policy.imitate(s)
        loss_a = self.multi_entropy_loss(a_weights, target_a)
        return loss_a
    
    def imitating(self, epoch):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)
        """
        self.policy.train()
        a_loss = 0.
        for i, data in enumerate(self.data_train):
            self.policy_optim.zero_grad()
            loss_a = self.imitate_loop(data)
            a_loss += loss_a.item()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
            self.policy_optim.step()
            
            if (i+1) % self.print_per_batch == 0:
                a_loss /= self.print_per_batch
                logging.debug('<<dialog policy>> epoch {}, iter {}, loss_a:{}'.format(epoch, i, a_loss))
                a_loss = 0.
        
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch, True)
        self.policy.eval()

    def teacher_forcing(self, epoch):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)
        """
        if epoch%4!=0:
            return
        # self.policy.train()
        a_loss = 0.
        data = self.retrieve_expert(to_device_=False)
        self.policy_optim.zero_grad()
        loss_a = self.imitate_loop(data)
        a_loss += loss_a.item()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
        self.policy_optim.step()

    def imit_test(self, epoch, best):
        """
        provide an unbiased evaluation of the policy fit on the training dataset
        """        
        a_loss = 0.
        for i, data in enumerate(self.data_valid):
            loss_a = self.imitate_loop(data)
            a_loss += loss_a.item()
            
        a_loss /= len(self.data_valid)
        logging.debug('<<dialog policy>> validation, epoch {}, loss_a:{}'.format(epoch, a_loss))
        if a_loss < best:
            logging.info('<<dialog policy>> best model saved')
            best = a_loss
            self.save(self.save_dir, 'best', True)
            
        a_loss = 0.
        for i, data in enumerate(self.data_test):
            loss_a = self.imitate_loop(data)
            a_loss += loss_a.item()
            
        a_loss /= len(self.data_test)
        logging.debug('<<dialog policy>> test, epoch {}, loss_a:{}'.format(epoch, a_loss))
        return best
    
    def train_disc(self, epoch, batchsz):
        batch = self.sample(batchsz)
        self.rewarder.train_disc(batch, epoch)
    
    def test_disc(self, epoch, batchsz, best):
        batch = self.sample(batchsz)
        best = self.rewarder.test_disc(batch, epoch, best)
        return best

    def update(self, batchsz, epoch, best=None):
        """
        firstly sample batchsz items and then perform optimize algorithms.
        :param batchsz:
        :param epoch:
        :param best:
        :return:
        """
        backward = True if best is None else False
        if backward:
            self.policy.train()
            self.teacher_forcing(epoch)
        
        for p in self.rewarder.irl_params: # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update
        for _ in range(1):
            batch = self.sample(batchsz)
            s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
            a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
            next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
            d = torch.from_numpy(np.stack(batch.domain)).to(device=DEVICE)
            mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
            human_r = torch.FloatTensor(np.stack(batch.reward)).to(device=DEVICE)
            batchsz = s.size(0)
            
            # 2. update reward estimator
            inputs = (s, a, next_s, d)
            if backward:
                self.rewarder.update_disc(inputs, batchsz, epoch)
            else:
                best[1] = self.rewarder.update_disc(inputs, batchsz, epoch, best[1])


        # 5. update dialog policy
        for p in self.rewarder.irl_params:
                p.requires_grad = False # to avoid computation

        perm = torch.randperm(batchsz)
        # shuffle the variable for mutliple optimize
        s_shuf, a_shuf, d_shuf =  s[perm], a[perm], d[perm]

        # 2. get mini-batch for optimizing
        optim_chunk_num = int(np.ceil(batchsz / self.optim_batchsz))
        # chunk the optim_batch for total batch
        s_shuf, a_shuf, d_shuf = torch.chunk(s_shuf, optim_chunk_num), \
                                 torch.chunk(a_shuf, optim_chunk_num), \
                                 torch.chunk(d_shuf, optim_chunk_num)
        # 3. iterate all mini-batch to optimize
        policy_loss, value_loss, domain_loss_hist = 0., 0., 0.
        for s_b, a_b, d_b in zip(s_shuf, a_shuf, d_shuf):
            self.policy_optim.zero_grad()
            self.rewarder.irl_optim.zero_grad()  # clean the history gradisent in Disc  
            # logging.info(s_b.shape)
            if np.random.random()<0.5:
                s_b, _, _ = self.retrieve_expert()
            action = self.policy(s_b)
            disc_v = self.rewarder.estimate(s_b, action)
            if self.gan_type=='vanilla':
                real_labels = torch.ones_like(disc_v.view(-1)).detach()
                gen_loss = self.loss(disc_v.view(-1), real_labels)
            elif self.gan_type=='wgan':
                gen_loss = -disc_v.mean()
            policy_loss += gen_loss.item()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
            self.policy_optim.step()

        
        policy_loss /= optim_chunk_num
        # domain_loss_hist /= optim_chunk_num
        logging.debug('<<dialog policy>> epoch {}, policy, loss {}'.format(epoch, policy_loss))
        # logging.debug('<<dialog policy>> epoch {}, iteration {}, domain, loss {}'.format(epoch, i, domain_loss_hist))

        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
            with open(self.save_dir+'/'+str(epoch)+'.pkl', 'wb') as f:
                pickle.dump(best, f)
        self.policy.eval()

    def update_no_simulator(self, batchsz, epoch, best=None):
        """
        firstly sample batchsz items and then perform optimize algorithms.
        :param batchsz:
        :param epoch:
        :param best:
        :return:
        """
        backward = True if best is None else False
        if backward:
            self.policy.train()
            # self.teacher_forcing(epoch)
        
        for p in self.rewarder.irl_params: # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update
        for _ in range(self.disc_times):
            s, _, d = self.retrieve_expert()
            batchsz = s.size(0)
            fake_a = self.policy.select_action(s).view(batchsz, self.cfg.a_dim).detach()

            
            # 2. update reward estimator
            # the second state is just a placeholder
            inputs = (s, fake_a, s, d)
            if backward:
                self.rewarder.update_disc(inputs, batchsz, epoch)
            else:
                best[1] = self.rewarder.update_disc(inputs, batchsz, epoch, best[1])


        # 5. update dialog policy
        for p in self.rewarder.irl_params:
                p.requires_grad = False # to avoid computation

        perm = torch.randperm(batchsz)
        # shuffle the variable for mutliple optimize
        s_b = s[perm]
        # 3. iterate all mini-batch to optimize
        policy_loss, value_loss, domain_loss_hist = 0., 0., 0.

        self.policy_optim.zero_grad()
        self.rewarder.irl_optim.zero_grad()  # clean the history gradisent in Disc  

        action = self.policy(s_b)
        disc_v = self.rewarder.estimate(s_b, action)
        if self.gan_type=='vanilla':
            real_labels = torch.ones_like(disc_v.view(-1)).detach()
            gen_loss = self.loss(disc_v.view(-1), real_labels)
        elif self.gan_type=='wgan':
            gen_loss = -disc_v.mean()
        policy_loss += gen_loss.item()
        gen_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
        self.policy_optim.step()

        # domain_loss_hist /= optim_chunk_num
        logging.debug('<<dialog policy>> epoch {}, policy, loss {}'.format(epoch, policy_loss))
        # logging.debug('<<dialog policy>> epoch {}, iteration {}, domain, loss {}'.format(epoch, i, domain_loss_hist))

        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
            with open(self.save_dir+'/'+str(epoch)+'.pkl', 'wb') as f:
                pickle.dump(best, f)
        self.policy.eval()

    def sample(self, batchsz):
        """
        Given batchsz number of task, the batchsz will be splited equally to each processes
        and when processes return, it merge all data and return
        :param batchsz:
        :return: batch
        """

        # batchsz will be splitted into each process,
        # final batchsz maybe larger than batchsz parameters
        process_batchsz = np.ceil(batchsz / self.process_num).astype(np.int32)
        # buffer to save all data
        queue = mp.Queue()

        # start processes for pid in range(1, processnum)
        # if processnum = 1, this part will be ignored.
        # when save tensor in Queue, the process should keep alive till Queue.get(),
        # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
        # however still some problem on CUDA tensors on multiprocessing queue,
        # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
        # so just transform tensors into numpy, then put them into queue.
        evt = mp.Event()
        processes = []
        for i in range(self.process_num):
            process_args = (i, queue, evt, self.env_list[i], self.policy, process_batchsz, self.human_reward)
            processes.append(mp.Process(target=sampler, args=process_args))
        for p in processes:
            # set the process as daemon, and it will be killed once the main process is stoped.
            p.daemon = True
            p.start()

        # we need to get the first Memory object and then merge others Memory use its append function.
        pid0, buff0 = queue.get()
        for _ in range(1, self.process_num):
            pid, buff_ = queue.get()
            buff0.append(buff_) # merge current Memory into buff0
        evt.set()

        # now buff saves all the sampled data
        buff = buff0

        return buff.get_batch()
    
    def evaluate(self, save_dialog=False):
        self.policy.eval()
        if save_dialog:
            with open('./data/goal.json', 'r') as f:
                saved_goal_list=json.load(f)
        collected_dialog = []
        env = self.env_list[0]
        traj_len = 40
        reward_tot, turn_tot, inform_tot, match_tot, success_tot = [], [], [], [], []
        collect_goal_list = []
        for seed in range(1000):
            dialog_list =[]
            if save_dialog:
                s = env.reset(seed, saved_goal=saved_goal_list[seed])
            else:
                s = env.reset(seed, saved_goal=None)
            # collect_goal_list.append(env.goal.domain_goals_ori)
            # print('seed', seed)
            # logging.info('goal: {}'.format(env.goal.domain_goals))
            # logging.info('usr: {}'.format(s['user_action']))
            dialog_list.append(s['user_action'])
            turn = traj_len
            reward = []
            value = []
            mask = []
            for t in range(traj_len):
                s_vec = torch.Tensor(state_vectorize(s, env.cfg, env.db)).to(device=DEVICE)
                # mode with policy during evaluation
                a = self.policy.select_action(s_vec, False)
                next_s, done = env.step(s, a.cpu())
                next_s_vec = torch.Tensor(state_vectorize(next_s, env.cfg, env.db)).to(device=DEVICE)
                a_expand = id2onehot(a)
                r = self.rewarder.estimate(s_vec,a_expand.view(-1))
                reward.append(r.item())
                s = next_s
                # logging.info('sys: {}'.format(s['last_sys_action']))
                # logging.info('usr: {}'.format(s['user_action']))
                dialog_list.append(s['last_sys_action'])
                dialog_list.append(s['user_action'])

                if done:
                    mask.append(0)
                    turn = t+2 # one due to counting from 0, the one for the last turn
                    break
                mask.append(1)

            reward_tot.append(np.mean(reward))
            turn_tot.append(turn)
            match_tot += self.evaluator.match_rate(s)
            inform_tot.append(self.evaluator.inform_F1(s))
            reward = torch.Tensor(reward)
            mask = torch.LongTensor(mask)
            # print('turn', turn)
            match_session = self.evaluator.match_rate(s, True)
            # print('match', match_session)
            inform_session = self.evaluator.inform_F1(s, True)
            # print('inform', inform_session)
            if (match_session == 1 and inform_session[1] == 1) \
            or (match_session == 1 and inform_session[1] is None) \
            or (match_session is None and inform_session[1] == 1):
                # print('success', 1)
                success_tot.append(1)
            else:
                # print('success', 0)
                success_tot.append(0)
            dialog_dict={
                'goal id': seed,
                'goal': env.goal.domain_goals,
                'dialog': dialog_list,
                'turn': turn,
                'status': success_tot[-1]
            }
            collected_dialog.append(dialog_dict)
        
        logging.info('reward {}'.format(np.mean(reward_tot)))
        logging.info('turn {}'.format(np.mean(turn_tot)))
        logging.info('match {}'.format(np.mean(match_tot)))
        TP, FP, FN = np.sum(inform_tot, 0)
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        F1 = 2 * prec * rec / (prec + rec)
        logging.info('inform rec {}, F1 {}'.format(rec, F1))
        logging.info('success {}'.format(np.mean(success_tot)))
        self.policy.train()
        # with open("./data/goal.json",'w') as f:
        #     json.dump(collect_goal_list, f, indent=4)
        if save_dialog:
            self.save_dialog(self.save_dir, collected_dialog)

    def save_dialog(self, directory, collected_dialog):
        if not os.path.exists(directory):
            os.makedirs(directory)
        des_path = directory + '/' + 'collected_dialog.json'
        with open(des_path, 'w') as f:
            json.dump(collected_dialog, f, indent=4)

    def expert_generator(self):
        env = self.env_list[0]
        traj_len = 40
        reward_tot, turn_tot, inform_tot, match_tot, success_tot = [], [], [], [], []
        success_dialog = []
        while len(success_dialog)<10000:
            seed = np.random.randint(2000000)
            s = env.reset(seed)
            print('seed', seed)
            print('goal', env.goal.domain_goals)
            print('usr', s['user_action'])
            turn = traj_len
            reward = []
            value = []
            mask = []
            dialog_turn = []
            for t in range(traj_len):
                s_vec = torch.Tensor(state_vectorize(s, env.cfg, env.db)).to(device=DEVICE)
                # mode with policy during evaluation
                a = self.policy.select_action(s_vec, False)
                next_s, done = env.step(s, a.cpu())
                next_s_vec = torch.Tensor(state_vectorize(next_s, env.cfg, env.db)).to(device=DEVICE)
                r = self.reward_human(s, done)
                pair = (s_vec, a, next_s_vec, r, done)
                dialog_turn.append(copy.deepcopy(pair))
                s = next_s
                if done:
                    mask.append(0)
                    turn = t+2 # one due to counting from 0, the one for the last turn
                    break
            if r > 0:
                success_dialog += dialog_turn
                logging.info("success dialog: {}".format(len(success_dialog)))
        torch.save(success_dialog, './data/expert_dialog_art.pt')



        
    def check_success(self, s):
        match_session = self.evaluator.match_rate(s, True)
        inform_session = self.evaluator.inform_F1(s, True)
        if (match_session == 1 and inform_session[1] == 1) \
        or (match_session == 1 and inform_session[1] is None) \
        or (match_session is None and inform_session[1] == 1):
            return True
        else:
            return False
    
    def reward_human(self, s, done):
        success = self.check_success(s)
        if success:
            reward = 0.2 * 40 
        if not success and not done:
            reward = -0.1
        if not success and done:
            reward = -0.1 * 40  # 10
        return reward

    def save(self, directory, epoch, rl_only=False):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        if not rl_only:
            self.rewarder.save_irl(directory, epoch)

        torch.save(self.value.state_dict(), directory + '/' + str(epoch) + '_ppo.val.mdl')
        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_ppo.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

    def load(self, filename):
        self.rewarder.load_irl(filename)
        # if self.realtrain:
            # filename='model_saved_ori/model_agenda_pre_ori/best'
            # filename='model_saved/model_agenda_pre_mt_op_1.0/best'
        value_mdl = filename + '_ppo.val.mdl'
        policy_mdl = filename + '_ppo.pol.mdl'
        if os.path.exists(value_mdl):
            self.value.load_state_dict(torch.load(value_mdl))
            logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(value_mdl))
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl))
            logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))
        
        best_pkl = filename + '.pkl'
        if os.path.exists(best_pkl):
            with open(best_pkl, 'rb') as f:
                best = pickle.load(f)
        else:
            best = [float('inf'),float('inf'),float('-inf')]
        return best

    
class MultiDiscretePolicy(nn.Module):
    def __init__(self, cfg, feature_extractor=None):
        super(MultiDiscretePolicy, self).__init__()
        self.cfg = cfg
        self.argmax_type = cfg.argmax_type
        self.feature_extractor = feature_extractor
        self.net = nn.Sequential(nn.Linear(cfg.s_dim, cfg.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(cfg.h_dim, cfg.a_dim//4),
                                #  nn.ReLU()
                                 nn.LeakyReLU(0.2, True)
                                 )

        self.gumble_length_index = self.gumble_index_multiwoz_binary()
        self.gumble_num = len(self.gumble_length_index)
        self.last_layers = nn.ModuleList()
        for gumbel_width in self.gumble_length_index:
            self.last_layers.append(nn.Linear(cfg.a_dim//4, gumbel_width))
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
            if self.argmax_type == 'soft':
                out = self.gumbel_connector.forward_ST(out.view(-1,  g_width), self.cfg.temperature)
            elif self.argmax_type =='gumbel':
                out = self.gumbel_connector.forward_ST_gumbel(out.view(-1,  g_width), self.cfg.temperature)
            else:
                raise ValueError("no such argmax type: {}".format(self.argmax_type))
            input_to_gumbel.append(out)
        action_rep = torch.cat(input_to_gumbel, -1)
        return action_rep
    
    def imitate(self, s):
        a_weights = self.net(s)
        input_to_gumbel = []
        for layer, g_width in zip(self.last_layers, self.gumble_length_index):
            out = layer(a_weights)
            input_to_gumbel.append(out)
        action_rep = torch.cat(input_to_gumbel, -1)
        return action_rep
        
    
    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: [a_dim]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        action = self.forward(s)
        action = onehot2id(action).view(-1)
        return action
    