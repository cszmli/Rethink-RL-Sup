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
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch import multiprocessing as mp
from utils import GumbelConnector, onehot2id, id2onehot
from rlmodule import Value, Memory, Transition
from estimator import RewardEstimator
from disc_estimator import DiscEstimator
from utils import state_vectorize, to_device, domain_vectorize, summary, cast_type
from metrics import Evaluator
from decoders import DecoderRNN, GEN, TEACH_FORCE
import criterions
from utils import INT, FLOAT, LONG
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = True if torch.cuda.is_available() else False
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


class DiaSeq(object):
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
        self.policy_clip = args.policy_clip
        self.train_direc = args.train_direc
        self.realtrain = realtrain
        self.process_num = process_num
        self.human_reward = human_reward(args, cfg)
        self.gan_type = args.gan_type
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
        
        # if pre:
        self.print_per_batch = args.print_per_batch
        from dbquery import DBQuery
        db = DBQuery(args.data_dir)
        self.data_train = manager.create_dataset_seq('train', args.batchsz, cfg, db)
        self.data_valid = manager.create_dataset_seq('valid', args.batchsz, cfg, db)
        self.data_test = manager.create_dataset_seq('test', args.batchsz, cfg, db)
        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()
        self.evaluator = Evaluator(args.data_dir, cfg)

        # else:
        #     # self.rewarder = RewardEstimator(args, manager, cfg, pretrain=pre_irl, inference=infer, feature_extractor=self.policy)
        #     self.rewarder = DiscEstimator(args, manager, cfg, pretrain=pre_irl, inference=infer)
        #     self.evaluator = Evaluator(args.data_dir, cfg)
        #     from dbquery import DBQuery
        #     db = DBQuery(args.data_dir)
        #     self.data_train = manager.create_dataset_seq('train', args.batchsz, cfg, db)
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
        self.epoch = 0
        self.best_valid_loss = np.inf
        self.valid_loss_threshold = np.inf
        self.patience = 10
    
    def retrieve_expert(self, to_device_=True):
        try:
            data = self.expert_iter.next()
        except StopIteration:
            self.expert_iter = iter(self.data_train)
            data = self.expert_iter.next()
            self.epoch += 1
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

    def prepare_data(self, data_batch):
        s_real, a_real_, next_s_real, d_real = to_device(data_batch)

    
    def update_loop(self, data):
        # gen_type: beam, greedy
        # mode: GEN, TEACH_FORCE
        # s, a, d, a_seq = self.retrieve_expert(True)
        s, _, _, a_seq = data
        s = s.to(device=DEVICE)
        a_seq = a_seq.to(device=DEVICE)
        loss = self.policy(s=s, a_seq=a_seq[:, :self.cfg.max_len], mode=TEACH_FORCE)
        return loss
    
    def train(self, epoch):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)
        """
        self.policy.train()
        a_loss = 0.
        for i, data in enumerate(self.data_train):
            self.policy_optim.zero_grad()
            loss_a = self.update_loop(data)
            a_loss += loss_a.item()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.policy_clip)
            self.policy_optim.step()
            
            if (i+1) % self.print_per_batch == 0:
                a_loss /= self.print_per_batch
                logging.debug('<<dialog policy>> epoch {}, iter {}, loss_a:{}'.format(epoch, i, a_loss))
                a_loss = 0.

        self.policy.eval()
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch, True)

        for i, data in enumerate(self.data_valid):
            loss_a = self.update_loop(data)
            a_loss += loss_a.item()
        valid_loss = a_loss/len(self.data_valid)
        logging.debug('<<dialog policy>> validation, epoch {}, loss_a:{}'.format(epoch, valid_loss))
        if valid_loss < self.best_valid_loss:
            if valid_loss <= self.valid_loss_threshold * self.cfg.improve_threshold:
                self.patience = max(self.patience,
                                epoch * self.cfg.patient_increase)
                self.valid_loss_threshold = valid_loss
                logging.info("Update patience to {}".format(self.patience))
            self.best_valid_loss = valid_loss

            logging.info('<<dialog policy>> best model saved')
            self.save(self.save_dir, 'best', True)
        

        if self.cfg.early_stop and self.patience <= epoch:
            if epoch < self.cfg.max_epoch:
                logging.info("!!Early stop due to run out of patience!!")

            logging.info("Best validation loss %f" % self.best_valid_loss)
            return True
        return False



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
        for seed in range(1000):            
            dialog_list =[]
            if save_dialog:
                s = env.reset(seed, saved_goal=saved_goal_list[seed])
            else:
                s = env.reset(seed, saved_goal=None)
            # print('seed', seed)
            # print('goal', env.goal.domain_goals)
            # print('usr', s['user_action'])
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
                r = self.human_reward.reward_human(next_s, done)
                reward.append(r)
                s = next_s                
                dialog_list.append(s['last_sys_action'])
                dialog_list.append(s['user_action'])
                # print('sys', s['last_sys_action'])
                # print('usr', s['user_action'])
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
        
        if save_dialog:
            self.save_dialog(self.save_dir, collected_dialog)

    def save_dialog(self, directory, collected_dialog):
        if not os.path.exists(directory):
            os.makedirs(directory)
        des_path = directory + '/' + 'collected_dialog.json'
        with open(des_path, 'w') as f:
            json.dump(collected_dialog, f, indent=4)


    def save(self, directory, epoch, rl_only=False):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # if not rl_only:
            # self.rewarder.save_irl(directory, epoch)

        torch.save(self.value.state_dict(), directory + '/' + str(epoch) + '_ppo.val.mdl')
        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_ppo.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

    def load(self, filename):
        # self.rewarder.load_irl(filename)
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
        self.test_gentype = cfg.test_gentype
        self.argmax_type = cfg.argmax_type
        self.feature_extractor = feature_extractor
        self.decoder_hidden = cfg.h_dim//2
        if self.cfg.activate_type == 'tanh':
            self.net = nn.Sequential(nn.Linear(cfg.s_dim, cfg.h_dim),
                                    nn.ReLU(),
                                    nn.Linear(cfg.h_dim, self.decoder_hidden),
                                    nn.Tanh()
                                    )
        elif self.cfg.activate_type == 'relu':
            self.net = nn.Sequential(nn.Linear(cfg.s_dim, cfg.h_dim),
                                    nn.ReLU(),
                                    nn.Linear(cfg.h_dim, self.decoder_hidden),
                                     nn.ReLU(),
                                    )
        else: 
            self.net = nn.Sequential(nn.Linear(cfg.s_dim, cfg.h_dim),
                                    nn.ReLU(),
                                    nn.Linear(cfg.h_dim, self.decoder_hidden),
                                    )
        if self.cfg.cat_mlp:
            decoder_input_size = cfg.embed_size + self.decoder_hidden
        else:
            decoder_input_size = cfg.embed_size
        self.go_id = 1
        self.eos_id = 2
        self.embedding = nn.Embedding(cfg.a_dim, cfg.embed_size,
                                      padding_idx=0)
        self.decoder = DecoderRNN(cfg.a_dim, cfg.max_len,
                            decoder_input_size, self.decoder_hidden,
                            self.go_id, self.eos_id,
                            n_layers=1, rnn_cell='gru',
                            input_dropout_p=0.,
                            dropout_p=0.,
                            use_gpu=USE_GPU,
                            embedding=self.embedding,
                            cat_mlp=self.cfg.cat_mlp)
                            
        self.da2idx = cfg.da2idx
        self.max_length = cfg.max_len
        self.nll_loss = criterions.NLLEntropy(0, avg_type=self.cfg.avg_type)

    def forward(self, s=None, a_seq=None, mode=None, gen_type=None):
        a_seq = cast_type(a_seq, LONG, USE_GPU)
        batch_size = s.shape[0]
        # logging.info("s shape: {}".format(s.shape))
        dec_init_state = self.net(s).unsqueeze(0)
        # logging.info("h: {}, inp: {}".format(dec_init_state.shape, a_seq.shape))
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size, a_seq[:, 0:-1], dec_init_state,
                                            mode=mode, gen_type=gen_type,
                                            beam_size=1)
        labels = a_seq[:, 1:].contiguous()
        enc_dec_nll = self.nll_loss(dec_outs, labels)
        return enc_dec_nll

    
    def select_action(self, s, sample=True):
        # here we repeat the state twice to avoid potential errors in Decoder
        s = s.view(-1, self.cfg.s_dim).repeat(2,1)
        batch_size = s.shape[0]
        dec_init_state = self.net(s).unsqueeze(0)
        a_seq = None
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size, a_seq, dec_init_state,
                                            mode=GEN, gen_type=self.test_gentype,
                                            beam_size=5)
        pred_labels = [t.cpu().data.numpy() for t in dec_ctx[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        pred_labels = pred_labels[0]
        act = torch.zeros(self.cfg.a_dim)
        for x in pred_labels:
            if x not in [0, 1, 2, 169]:
                act[x]=1.
            elif x == 2:
                break
        # logging.info(act)
        return act



class DecoderRNN_s(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN_s, self).__init__()
        self.hidden_size = hidden_size
        self.criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size, output_size, bias=False)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward_step(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def forward(self, mlp_hidden=None, trg_mask=None, max_len=20):
        decoder_states = [] 
        hidden = mlp_hidden
        for i in range(max_len):
            prev_embd = self.embedding[:, i].unsqueeze(1)
            output, hidden = self.forward_step(prev_embd, hidden)
            decoder_states.append(output)
        out_put = torch.cat(decoder_states, dim=1)
        return out_put
        # return F.log_softmax(self.proj(x), dim=-1)
    def loss(self, x, y):
        x = F.log_softmax(self.proj(x), dim=-1)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        return loss
    