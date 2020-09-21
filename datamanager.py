# -*- coding: utf-8 -*-
"""
/////
"""
import os
import json
import logging
import torch
from collections import defaultdict
import torch.utils.data as data
from copy import deepcopy
from utils import init_session, init_goal, state_vectorize, action_vectorize, domain_vectorize, action_seq, count_act
import numpy as np
def expand_da(meta):
    for k, v in meta.items():
        domain, intent = k.split('-')
        if intent.lower() == "request":
            for pair in v:
                pair.insert(1, '?')
        else:
            counter = {}
            for pair in v:
                if pair[0] == 'none':
                    pair.insert(1, 'none')
                else:
                    if pair[0] in counter:
                        counter[pair[0]] += 1
                    else:
                        counter[pair[0]] = 1
                    pair.insert(1, str(counter[pair[0]]))

class DataManager():
    """Offline data manager"""
    
    def __init__(self, data_dir, cfg):
        self.data = {}
        self.goal = {}
        
        self.data_dir_new = data_dir + '/processed_data'
        if os.path.exists(self.data_dir_new):
            logging.info('Load processed data file')
            for part in ['train','valid','test']:
                with open(self.data_dir_new + '/' + part + '.json', 'r') as f:
                    self.data[part] = json.load(f)
                with open(self.data_dir_new + '/' + part + '_goal.json', 'r') as f:
                    self.goal[part] = json.load(f)
        else:
            from dbquery import DBQuery
            db = DBQuery(data_dir)
            logging.info('Start preprocessing the dataset')
            self._build_data(data_dir, self.data_dir_new, cfg, db)
            
    def _build_data(self, data_dir, data_dir_new, cfg, db):
        data_filename = data_dir + '/' + cfg.data_file
        with open(data_filename, 'r') as f:
            origin_data = json.load(f)
        
        for part in ['train','valid','test']:
            self.data[part] = []
            self.goal[part] = {}
            
        valList = []
        with open(data_dir + '/' + cfg.val_file) as f:
            for line in f:
                valList.append(line.split('.')[0])
        testList = []
        with open(data_dir + '/' + cfg.test_file) as f:
            for line in f:
                testList.append(line.split('.')[0])
            
        for k_sess in origin_data:
            sess = origin_data[k_sess]
            if k_sess in valList:
                part = 'valid'
            elif k_sess in testList:
                part = 'test'
            else:
                part = 'train'
            turn_data, session_data = init_session(k_sess, cfg)
            init_goal(session_data, sess['goal'], cfg)
            self.goal[part][k_sess] = session_data
            belief_state = turn_data['belief_state']
            
            for i, turn in enumerate(sess['log']):
                turn_data['others']['turn'] = i
                turn_data['others']['terminal'] = i + 2 >= len(sess['log'])
                da_origin = turn['dialog_act']
                expand_da(da_origin)
                turn_data['belief_state'] = deepcopy(belief_state) # from previous turn

                if i % 2 == 0: # user
                    if 'last_sys_action' in turn_data:
                        turn_data['history']['sys'] = dict(turn_data['history']['sys'], **turn_data['last_sys_action'])
                        del(turn_data['last_sys_action'])
                    turn_data['last_user_action'] = deepcopy(turn_data['user_action'])
                    turn_data['user_action'] = dict()
                    for domint in da_origin:
                        domain_intent = da_origin[domint]
                        _domint = domint.lower()
                        _domain, _intent = _domint.split('-')
                        if _intent == 'thank':
                            _intent = 'welcome'
                            _domint = _domain+'-'+_intent
                        for slot, p, value in domain_intent:
                            _slot = slot.lower()
                            _value = value.strip()
                            _da = '-'.join((_domint, _slot, p))
                            if _da in cfg.da_usr:
                                turn_data['user_action'][_da] = _value
                                if _intent == 'inform':
                                    inform_da = _domain+'-'+_slot+'-1'
                                    if inform_da in cfg.inform_da:
                                        belief_state['inform'][_domain][_slot] = _value
                                elif _intent == 'request':
                                    request_da = _domain+'-'+_slot
                                    if request_da in cfg.request_da:
                                        belief_state['request'][_domain].add(_slot)
                        
                else: # sys
                    if 'last_user_action' in turn_data:
                        turn_data['history']['user'] = dict(turn_data['history']['user'], **turn_data['last_user_action'])
                        del(turn_data['last_user_action'])
                    turn_data['last_sys_action'] = deepcopy(turn_data['sys_action'])
                    turn_data['sys_action'] = dict()
                    for domint in da_origin:
                        domain_intent = da_origin[domint]
                        _domint = domint.lower()
                        _domain, _intent = _domint.split('-')
                        for slot, p, value in domain_intent:
                            _slot = slot.lower()
                            _value = value.strip()
                            _da = '-'.join((_domint, _slot, p))
                            if _da in cfg.da:
                                turn_data['sys_action'][_da] = _value
                                if _intent == 'inform' and _domain in belief_state['request']:
                                    belief_state['request'][_domain].discard(_slot)
                                elif _intent == 'book' and _slot == 'ref':
                                    for domain in belief_state['request']:
                                        if _slot in belief_state['request'][domain]:
                                            belief_state['request'][domain].remove(_slot)
                                            break
                           
                    book_status = turn['metadata']
                    for domain in cfg.belief_domains:
                        if book_status[domain]['book']['booked']:
                            entity = book_status[domain]['book']['booked'][0]
                            if domain == 'taxi':
                                belief_state['booked'][domain] = 'booked'
                            elif domain == 'train':
                                found = db.query(domain, [('trainID', entity['trainID'])])
                                belief_state['booked'][domain] = found[0]['ref']
                            else:
                                found = db.query(domain, [('name', entity['name'])])
                                belief_state['booked'][domain] = found[0]['ref']        
                
                if i + 1 == len(sess['log']):
                    turn_data['next_belief_state'] = belief_state
                
                self.data[part].append(deepcopy(turn_data))
                
        def _set_default(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError
        os.makedirs(data_dir_new)
        for part in ['train','valid','test']:
            with open(data_dir_new + '/' + part + '.json', 'w') as f:
                self.data[part] = json.dumps(self.data[part], default=_set_default)
                f.write(self.data[part])
                self.data[part] = json.loads(self.data[part])
            with open(data_dir_new + '/' + part + '_goal.json', 'w') as f:
                self.goal[part] = json.dumps(self.goal[part], default=_set_default)
                f.write(self.goal[part])
                self.goal[part] = json.loads(self.goal[part])
        
    def create_dataset(self, part, file_dir, cfg, db):
        datas = self.data[part]
        goals = self.goal[part]
        s, a, next_s, a_seq = [], [], [], []
        d_m = []
        for idx, turn_data in enumerate(datas):
            if turn_data['others']['turn'] % 2 == 0:
                continue
            turn_data['user_goal'] = goals[turn_data['others']['session_id']]
            s.append(torch.Tensor(state_vectorize(turn_data, cfg, db, True)))
            a.append(torch.Tensor(action_vectorize(turn_data, cfg)))
            d_m.append(torch.Tensor(domain_vectorize(turn_data, cfg)))
            a_seq.append(torch.Tensor(action_seq(turn_data, cfg)))

            if not int(turn_data['others']['terminal']):
                next_s.append(torch.Tensor(state_vectorize(datas[idx+2], cfg, db, True)))
            else:
                next_turn_data = deepcopy(turn_data)
                next_turn_data['others']['turn'] = -1
                next_turn_data['user_action'] = {}
                next_turn_data['last_sys_action'] = next_turn_data['sys_action']
                next_turn_data['sys_action'] = {}
                next_turn_data['belief_state'] = next_turn_data['next_belief_state']
                next_s.append(torch.Tensor(state_vectorize(next_turn_data, cfg, db, True)))
        torch.save((s, a, next_s, d_m, a_seq), file_dir)
        
    def create_dataset_rl(self, part, batchsz, cfg, db):
        logging.debug('start loading rl {}'.format(part))
        if cfg.data_ratio==100:
            file_dir = self.data_dir_new + '/' + part + '.pt'
        else:
            file_dir = self.data_dir_new + '/' + part + '.{}.pt'.format(cfg.data_ratio)
        # file_dir = self.data_dir_new + '/' + part + '.pt'  # pt does not have act_seq
        if not os.path.exists(file_dir):
            self.create_dataset(part, file_dir, cfg, db)
        
        s, a, x, d = torch.load(file_dir)
        ds_all = len(s)
        # for ratio in [10,40,70]:
        #     ds = int(ds_all * ratio * 0.01)
        #     x1, x2, x3, x4 = s[:ds], a[:ds], x[:ds], d[:ds]
        #     file_dir_x = self.data_dir_new + '/' + part + '.{}.pt'.format(ratio)
        #     torch.save((x1, x2, x3, x4), file_dir_x)

        dataset = DatasetRL(s, a, d)
        dataloader = data.DataLoader(dataset, batchsz, True)
        logging.debug('finish loading rl {}'.format(part))
        return dataloader
    
    def create_dataset_seq(self, part, batchsz, cfg, db):
        logging.debug('start loading seq {}'.format(part))
        if cfg.data_ratio==100:
            file_dir = self.data_dir_new + '/' + part + '.unk.pt'
        else:
            file_dir = self.data_dir_new + '/' + part + '.{}.unk.pt'.format(cfg.data_ratio)
        # file_dir = self.data_dir_new + '/' + part + '.unk.pt'
        # file_dir = self.data_dir_new + '/' + part + '.pt'
        if not os.path.exists(file_dir):
            self.create_dataset(part, file_dir, cfg, db)
        
        s, a, x, d, a_seq = torch.load(file_dir)
        ds_all = len(s)
        logging.info("data size: {}".format(ds_all))
        # for ratio in [10,40,70]:
        #     ds = int(ds_all * ratio * 0.01)
        #     x1, x2, x3, x4, x5 = s[:ds], a[:ds], x[:ds], d[:ds], a_seq[:ds]
        #     file_dir_x = self.data_dir_new + '/' + part + '.{}.unk.pt'.format(ratio)
        #     torch.save((x1, x2, x3, x4, x5), file_dir_x)


        # count_act(a_seq, cfg)
        dataset = DatasetSeq(s, a, d, a_seq)
        dataloader = data.DataLoader(dataset, batchsz, True)
        logging.debug('finish loading seq {}'.format(part))
        return dataloader
    
    def create_dataset_irl(self, part, batchsz, cfg, db):
        logging.debug('start loading irl {}'.format(part))
        if cfg.data_ratio==100:
            file_dir = self.data_dir_new + '/' + part + '.pt'
        else:
            file_dir = self.data_dir_new + '/' + part + '.{}.pt'.format(cfg.data_ratio)
        if not os.path.exists(file_dir):
            self.create_dataset(part, file_dir, cfg, db)
        
        s, a, next_s, d= torch.load(file_dir)
        dataset = DatasetIrl(s, a, next_s, d)
        dataloader = data.DataLoader(dataset, batchsz, True)
        logging.debug('finish loading irl {}'.format(part))
        return dataloader
    


class DatasetRL(data.Dataset):
    def __init__(self, s_s, a_s, d_s):
        self.s_s = s_s
        self.a_s = a_s
        self.d_s = d_s
        self.num_total = len(s_s)
    
    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        d = self.d_s[index]
        return s, a, d
    
    def __len__(self):
        return self.num_total


class DatasetSeq(data.Dataset):
    def __init__(self, s_s, a_s, d_s, a_seq):
        self.s_s = s_s
        self.a_s = a_s
        self.d_s = d_s
        self.a_seq = a_seq
        self.num_total = len(s_s)
        a = np.zeros(20)
        a_list = defaultdict(int)
        # for a_l in a_s:
        #     a[int(a_l.sum().item())]+=1
        #     key = str(a_l.tolist())
        #     a_list[key] += 1
        # k=0
        # for w in sorted(a_list, key=a_list.get, reverse=True):
        #     logging.info("{}: {}".format(w, a_list[w]))
        #     logging.info(k)
        #     k+=1
        # logging.info(a)
        # logging.info(len(a_list))
    
    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        d = self.d_s[index]
        a_seq = self.a_seq[index]
        return s, a, d, a_seq
    
    def __len__(self):
        return self.num_total   
    


class DatasetIrl(data.Dataset):
    def __init__(self, s_s, a_s, next_s_s, d_s):
        self.s_s = s_s
        self.a_s = a_s
        self.next_s_s = next_s_s
        self.d_s = d_s
        self.num_total = len(s_s)
    
    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        next_s = self.next_s_s[index]
        d = self.d_s[index]
        return s, a, next_s, d
    
    def __len__(self):
        return self.num_total   
    
