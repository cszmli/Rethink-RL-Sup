# -*- coding: utf-8 -*-
"""
/////
"""
import sys
import time
from datetime import datetime
import os
import logging
from utils import get_parser, init_logging_handler, summary
from datamanager import DataManager
from user import UserNeural
from usermanager import UserDataManager
from agenda import UserAgenda
from gan import GAN
from config import MultiWozConfig
from torch import multiprocessing as mp
import torch.nn.functional as F

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
    agent = GAN(make_env, args, manager, config, args.process, pre_irl=True)
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
    parser = get_parser()
    argv = sys.argv[1:]
    args, _ = parser.parse_known_args(argv)
    if not args.load_user:
        args.load_user = args.save_dir+'/best'
    
    if args.config == 'multiwoz':
        config = MultiWozConfig()
        config.s_lambda = args.s_lambda
        config.temperature = args.temperature
        config.argmax_type = args.argmax_type
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
        # dir_name = datetime.now().isoformat()
        args.save_dir = os.path.join('model_'+args.gan_type, current_time)
        logging.info(args.save_dir)

        agent = GAN(make_env, args, manager, config, 1, infer=True)
        agent.load(args.load)
        logging.info("model loading finish and start evaluating")
        agent.evaluate(save_dialog=True)
        # agent.evaluate()
        # agent.expert_generator()
        
    else: # training
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # dir_name = datetime.now().isoformat()
        args.save_dir = os.path.join('model_'+args.gan_type, current_time)
        logging.info(args.save_dir)
        
        # current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        logging.debug('train {}'.format(current_time))
    
        agent = GAN(make_env, args, manager, config, args.process, realtrain=True)
        best = agent.load(args.load)
        # logging.info("model loading finish and start evaluating")
        # agent.evaluate()
        # irl, rl
        for i in range(args.epoch):
            if i%4==0:
                agent.evaluate()
            if args.rl_sim == 'yes':
                agent.update(args.batchsz_traj, i)
                best = agent.update(args.batchsz, i, best)
            else:
                agent.update_no_simulator(args.batchsz_traj, i)
                best = agent.update_no_simulator(args.batchsz, i, best)

            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.debug('epoch {} {}'.format(i, current_time))

        logging.info("############## Start Evaluating ##############")
        agent_test = GAN(make_env, args, manager, config, 1, infer=True)
        agent_test.load(args.save_dir+'/best')
        logging.info("model loading finish and start evaluating")
        agent_test.evaluate()
