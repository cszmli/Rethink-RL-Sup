# -*- coding: utf-8 -*-
"""
/////
"""
import sys
import time
from datetime import datetime
import logging
from utils import get_parser, init_logging_handler, summary
from datamanager import DataManager
from user import UserNeural
from usermanager import UserDataManager
from agenda import UserAgenda
from mlp2seq import DiaSeq
from config import MultiWozConfig, SeqWozConfig
from torch import multiprocessing as mp
import torch.nn.functional as F
import random
import os
import torch
import numpy as np

def worker_user(args, manager, config):
    init_logging_handler(args.log_dir, '_user')
    env = UserNeural(args, manager, config, True)
    
    best = float('inf')
    for e in range(args.epoch):
        env.imitating(e)
        best = env.imit_test(e, best)

def worker_policy(args, manager, config):
    init_logging_handler(args.log_dir, '_policy')
    agent = GAN(None, args, manager, config, 0, pre=True)
    
    best = float('inf')
    for e in range(args.epoch):
        agent.imitating(e)
        best = agent.imit_test(e, best)

def worker_estimator(args, manager, config, make_env):
    init_logging_handler(args.log_dir, '_estimator')
    agent = DiaSeq(make_env, args, manager, config, args.process, pre_irl=True)
    agent.load(args.save_dir+'/best')
    
    best0, best1 = float('inf'), float('inf')
    for e in range(args.epoch):
        agent.train_disc(e, args.batchsz_traj)
        best0 = agent.test_disc(e, args.batchsz, best0)

def make_env_neural():
    env = UserNeural(args, usermanager, config)
    env.load(args.load_user)
    return env

def make_env_agenda():
    env = UserAgenda(args.data_dir, config)
    return env

if __name__ == '__main__':
    manualSeed = random.randrange(99999)
    # manualSeed = 44245
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    print("Random Seed: {}".format(manualSeed))

    parser = get_parser()
    argv = sys.argv[1:]
    args, _ = parser.parse_known_args(argv)

    dir_name = datetime.now().isoformat()
    args.save_dir = os.path.join('model_slurm', dir_name)

    if not args.load_user:
        args.load_user = args.save_dir+'/best'
    
    if args.config == 'multiwoz':
        config = SeqWozConfig()
        config.s_lambda = args.s_lambda
        config.temperature = args.temperature
        config.argmax_type = args.argmax_type
        config.cat_mlp = args.cat_mlp
        config.test_gentype = args.gen_type
        config.avg_type = args.avg_type
        config.activate_type = args.activate_type
        config.max_epoch = args.epoch
        config.early_stop = args.early_stop
        config.patient_increase = args.patient_increase
        config.improve_threshold = args.improve_threshold
        config.data_ratio = args.data_ratio
    else:
        raise NotImplementedError('Config of the dataset {} not implemented'.format(args.config))
    if args.simulator == 'neural':
        usermanager = UserDataManager(args.data_dir, config.data_file)
        make_env = make_env_neural
    elif args.simulator == 'agenda':
        make_env = make_env_agenda
    else:
        raise NotImplementedError('User simulator {} not implemented'.format(args.simulator))
    init_logging_handler(args.log_dir)
    logging.debug(str(args))
    
    manager = DataManager(args.data_dir, config)
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    if args.pretrain:
        logging.debug('pretrain')
        
        processes = []
        process_args = (args, manager, config)
        processes.append(mp.Process(target=worker_policy, args=process_args))
        if args.simulator == 'neural':
            process_args_user = (args, usermanager, config)
            processes.append(mp.Process(target=worker_user, args=process_args_user))
        for p in processes:
            p.start()
        
        for p in processes:
            p.join()
        worker_estimator(args, manager, config, make_env)
        logging.info("pretrain ends here")
            
    elif args.test:
        logging.debug('test')
        logging.disable(logging.DEBUG)
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        args.save_dir = os.path.join('model_seq', current_time)
        logging.info(args.save_dir)

        agent = DiaSeq(make_env, args, manager, config, 1, infer=True)
        agent.load(args.load)
        logging.info("model loading finish and start evaluating")
        agent.evaluate(save_dialog=True)
        # agent.expert_generator()
        
    else: # training
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # dir_name = datetime.now().isoformat()
        args.save_dir = os.path.join('model_'+args.gan_type, current_time)
        logging.info(args.save_dir)
        
        # current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        logging.debug('train {}'.format(current_time))
    
        agent = DiaSeq(make_env, args, manager, config, args.process, realtrain=True)
        # best = agent.load(args.load)
        # logging.info("model loading finish and start evaluating")
        # agent.evaluate()
        # irl, rl
        for i in range(args.epoch):
            if i%4==1:
                agent.evaluate()
            f = agent.train(i)
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.debug('epoch {} {}'.format(i, current_time))
            if f:
                break

        logging.info("############## Start Evaluating ##############")
        agent_test = DiaSeq(make_env, args, manager, config, 1, infer=True)
        agent_test.load(args.save_dir+'/best')
        logging.info("model loading finish and start evaluating")
        agent_test.evaluate()
