"""
template NLG for multiwoz dataset. templates are in `multiwoz_template_nlg/` dir.
See `example` function in this file for usage.
"""
import json
import os
import random
from pprint import pprint
import sys
import csv

import copy

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# supported slot
slot2word = {
    'Fee': 'fee',
    'Addr': 'address',
    'Area': 'area',
    'Stars': 'stars',
    'Internet': 'Internet',
    'Department': 'department',
    'Choice': 'choice',
    'Ref': 'reference number',
    'Food': 'food',
    'Type': 'type',
    'Price': 'price range',
    'Stay': 'stay',
    'Phone': 'phone',
    'Post': 'postcode',
    'Day': 'day',
    'Name': 'name',
    'Car': 'car type',
    'Leave': 'leave',
    'Time': 'time',
    'Arrive': 'arrive',
    'Ticket': 'ticket',
    'Depart': 'departure',
    'People': 'people',
    'Dest': 'destination',
    'Parking': 'parking',
    'Open': 'open',
    'Id': 'Id',
    'leaveAt': 'leaving time',
    'arriveBy': 'arrival time',
    'TrainID': 'TrainID'
}


class MultiwozTemplateNLG():
    def __init__(self, is_user):
        """
        :param is_user: if dialog_act from user or system
        both template are dict, *_template[dialog_act][slot] is a list of templates.
        """
        #super().__init__()
        self.is_user = is_user
        template_dir = os.path.dirname(os.path.abspath(__file__))
        # self.manual_user_template = read_json(os.path.join(template_dir, 'manual_user_template_nlg_clean.json'))
        # self.manual_system_template = read_json(os.path.join(template_dir, 'manual_system_template_nlg_clean.json'))
        self.manual_user_template = read_json(os.path.join('./data', 'manual_user_template_nlg_clean.json'))
        self.manual_system_template = read_json(os.path.join('./data', 'manual_system_template_nlg_clean.json'))

    def generate(self, dialog_acts):
        """
        NLG for Multiwoz dataset
        :param dialog_acts: {da1:[[slot1,value1],...], da2:...}
        :return: generated sentence
        """
        is_user = self.is_user
        if is_user:
            template = self.manual_user_template
        else:
            template = self.manual_system_template

        return self._manual_generate(dialog_acts, template)

    def _postprocess(self,sen):
        sen = sen.strip().capitalize()
        if len(sen) > 0 and sen[-1] != '?' and sen[-1] != '.':
            sen += '.'
        sen += ' '
        return sen

    def _manual_generate(self, dialog_acts, template):
        sentences = ''
        for dialog_act, slot_value_pairs in dialog_acts.items():
            intent = dialog_act.split('-')
            if 'Select'==intent[1]:
                slot2values = {}
                for slot, value in slot_value_pairs:
                    slot2values.setdefault(slot, [])
                    slot2values[slot].append(value)
                for slot, values in slot2values.items():
                    if slot == 'none': continue
                    sentence = 'Do you prefer ' + values[0]
                    for i, value in enumerate(values[1:]):
                        if i == (len(values) - 2):
                            sentence += ' or ' + value
                        else:
                            sentence += ' , ' + value
                    sentence += ' {} ? '.format(slot2word[slot])
                    sentences += sentence
            elif 'Request'==intent[1]:
                for slot, value in slot_value_pairs:
                    if dialog_act not in template or slot not in template[dialog_act]:
                        sentence = 'What is the {} of {} ? '.format(slot, dialog_act.split('-')[0].lower())
                        sentences += sentence
                    else:
                        sentence = random.choice(template[dialog_act][slot])
                        sentence = self._postprocess(sentence)
                        sentences += sentence
            elif 'general'==intent[0] and dialog_act in template:
                sentence = random.choice(template[dialog_act]['none'])
                sentence = self._postprocess(sentence)
                sentences += sentence
            else:
                for slot, value in slot_value_pairs:
                    if dialog_act in template and slot in template[dialog_act]:
                        sentence = random.choice(template[dialog_act][slot])
                        sentence = sentence.replace('#{}-{}#'.format(dialog_act.upper(), slot.upper()), str(value))
                    else:
                        if slot in slot2word:
                            sentence = 'The {} is {} . '.format(slot2word[slot], str(value))
                        else:
                            sentence = ''
                    sentence = self._postprocess(sentence)
                    sentences += sentence
        return sentences.strip()

def example(input_path, output_path):
    # dialog act
    dialog_acts = {}
    # whether from user or system

    template_dir = os.path.dirname(os.path.abspath(__file__))
    # R = read_json(input_path)
    #R = read_json(os.path.join(template_dir, input_path))
    # ACER_dialog = R[0]
    # goals = R[1]
    dialog_list = read_json(input_path)
    doc = open(output_path, 'w')

    t = 0

    # for dialogs in ACER_dialog:
    for dialogs in dialog_list:
        t = t+1
        doc.write("----------------------------------" + str(t) + "-------------------------------------------\n")
        #print("----------------------------------", t,"-------------------------------------------\n", file = doc)
        is_user = False
        # goal = goals[t-1]
        goal = dialogs['goal']
        dialog_history = dialogs['dialog']
        for dialog in dialog_history:
            #dialog is a dict
            is_user = not is_user

            dialog_acts = {}

            for slot, value in dialog.items():
                intent = slot.split('-')

                if is_user and intent[1] == 'inform' and value == 'none':
                    if intent[0] in goal:
                        if 'book' in goal[intent[0]]:
                            if intent[2] in goal[intent[0]]['book']:
                                value = goal[intent[0]]['book'][intent[2]]
                if value == 'none' and not intent[2] == 'none':
                    value = '#' + intent[0] + '-' + intent[1] + '-' + intent[2] + '#'

                if not intent[0] == 'general':
                    intent[0] = intent[0].capitalize()
                    intent[1] = intent[1].capitalize()
                domain = intent[0] + '-' + intent[1]

                if not intent[0] == 'general':
                    dialog_acts.setdefault(domain, [])
                    if not intent[2] == 'none':
                        intent[2] = intent[2].capitalize()
                    dialog_acts[domain].append([intent[2], value])

            for slot, value in dialog.items():
                intent = slot.split('-')
                if not intent[0] == 'general':
                    intent[0] = intent[0].capitalize()
                    intent[1] = intent[1].capitalize()
                domain = intent[0] + '-' + intent[1]
                if intent[0] == 'general':
                    dialog_acts.setdefault(domain, [])
                    if not intent[2] == 'none':
                        intent[2] = intent[2].capitalize()
                    dialog_acts[domain].append([intent[2], value])


            multiwoz_template_nlg = MultiwozTemplateNLG(is_user)
            # print(dialog_acts)
            # print(dialog_acts)
            # raise ValueError("dd")
            sentence = multiwoz_template_nlg.generate(dialog_acts)
            if sentence == '':
                continue

            if is_user:
                doc.write("usr: ")
                #print("usr: ", file = doc)
            else:
                doc.write("sys: ")
                #print("sys: ", file = doc)
            doc.write(sentence +"\n")
            #print(sentence +"\n", file = doc)

    doc.close()


def example_json(input_path, output_path):
    # dialog act
    dialog_acts = {}
    # whether from user or system

    template_dir = os.path.dirname(os.path.abspath(__file__))
    # R = read_json(input_path)
    #R = read_json(os.path.join(template_dir, input_path))
    # ACER_dialog = R[0]
    # goals = R[1]
    dialog_list = read_json(input_path)

    t = 0
    collected_dialog = []
    # for dialogs in ACER_dialog:
    for dialogs in dialog_list:
        t = t+1
        # doc.write("----------------------------------" + str(t) + "-------------------------------------------\n")

        #print("----------------------------------", t,"-------------------------------------------\n", file = doc)
        is_user = False
        # goal = goals[t-1]
        goal = dialogs['goal']
        goal_nlg = rewrite_goal(goal)
        dialog_history = dialogs['dialog']
        dialogs_copy = copy.deepcopy(dialogs)
        translated_dialog = []
        for dialog in dialog_history:
            #dialog is a dict
            is_user = not is_user
            dialog_acts = {}

            for slot, value in dialog.items():
                intent = slot.split('-')

                if is_user and intent[1] == 'inform' and value == 'none':
                    if intent[0] in goal:
                        if 'book' in goal[intent[0]]:
                            if intent[2] in goal[intent[0]]['book']:
                                value = goal[intent[0]]['book'][intent[2]]
                if value == 'none' and not intent[2] == 'none':
                    value = '#' + intent[0] + '-' + intent[1] + '-' + intent[2] + '#'

                if not intent[0] == 'general':
                    intent[0] = intent[0].capitalize()
                    intent[1] = intent[1].capitalize()
                domain = intent[0] + '-' + intent[1]

                if not intent[0] == 'general':
                    dialog_acts.setdefault(domain, [])
                    if not intent[2] == 'none':
                        intent[2] = intent[2].capitalize()
                    dialog_acts[domain].append([intent[2], value])

            for slot, value in dialog.items():
                intent = slot.split('-')
                if not intent[0] == 'general':
                    intent[0] = intent[0].capitalize()
                    intent[1] = intent[1].capitalize()
                domain = intent[0] + '-' + intent[1]
                if intent[0] == 'general':
                    dialog_acts.setdefault(domain, [])
                    if not intent[2] == 'none':
                        intent[2] = intent[2].capitalize()
                    dialog_acts[domain].append([intent[2], value])


            multiwoz_template_nlg = MultiwozTemplateNLG(is_user)
            # print(dialog_acts)
            sentence = multiwoz_template_nlg.generate(dialog_acts)
            if sentence == '':
                continue
            
            if is_user:
                translated_dialog.append('usr: ' + sentence)
                # doc.write("usr: ")
                #print("usr: ", file = doc)
            else:
                translated_dialog.append('sys: ' + sentence)
                # doc.write("sys: ")
                #print("sys: ", file = doc)
            # doc.write(sentence +"\n")
            #print(sentence +"\n", file = doc)
        dialogs_copy['nlg'] = translated_dialog
        dialogs_copy['goal_nlg'] = goal_nlg
        collected_dialog.append(dialogs_copy)

    with open(output_path, 'w') as f:
        json.dump(collected_dialog, f, indent=4)    




def build_crowdsource_pair(name1, file1, name2, file2, starting_id=0):
    pair_name = name1 + '_' + name2
    f1_list = read_json(file1)[starting_id:]
    f2_list = read_json(file2)[starting_id:]

    merge_list = []
    counter = starting_id
    for d1, d2 in zip(f1_list, f2_list):
        if random.random()>0.5:
            order = name1 + '_' + name2
            dialog={
                'goal': d1['goal_nlg'],
                'dialog_1': ' #'+'<br> #'.join(d1['nlg']),
                'dialog_2': ' #'+'<br> #'.join(d2['nlg']),
                'order': order,
                'goal_id': counter
            }
        else:
            order = name2 + '_' + name1
            dialog={
                'goal': d1['goal_nlg'],
                'dialog_1': ' #'+'<br> #'.join(d2['nlg']),
                'dialog_2': ' #'+'<br> #'.join(d1['nlg']),
                'order': order,
                'goal_id': counter
            }
        merge_list.append(dialog)
        counter+=1
        if len(merge_list)==100:
            break


    with open('./data/' + pair_name +'_oldtracker.csv', 'w') as f:
        fnames = ['goal', 'dialog_1', 'dialog_2', 'order', 'goal_id']
        writer = csv.DictWriter(f, fieldnames=fnames) 
        writer.writeheader()
        writer.writerows(merge_list)


def build_crowdsource_pair_for_all(name1, file1, name2, file2, name3, file3):

     
    def extract_dia(merge_list, pair_name, f1_list, f2_list, name_1, name_2):
        counter = 0
        for d1, d2 in zip(f1_list, f2_list):
            if random.random()>0.5:
                order = pair_name + '@@@' + name_1 + '_' + name_2
                dialog={
                    'goal': d1['goal_nlg'],
                    'dialog_1': ' #'+'<br> #'.join(d1['nlg']),
                    'dialog_2': ' #'+'<br> #'.join(d2['nlg']),
                    'order': order,
                    'goal_id': counter
                }
            else:
                order = pair_name + '@@@' + name_2 + '_' + name_1
                dialog={
                    'goal': d1['goal_nlg'],
                    'dialog_1': ' #'+'<br> #'.join(d2['nlg']),
                    'dialog_2': ' #'+'<br> #'.join(d1['nlg']),
                    'order': order,
                    'goal_id': counter
                }
            merge_list.append(dialog)
            counter+=1
            if counter>=100:
                break
        return merge_list
    
    merge_list = []
    pair_name1 = name1 + '_' + name3
    f1_list = read_json(file1)
    f3_list = read_json(file3)
    merge_list = extract_dia(merge_list, pair_name1, f1_list, f3_list, name1, name3)
    

    pair_name2 = name2 + '_' + name3
    f2_list = read_json(file2)
    merge_list = extract_dia(merge_list, pair_name2, f2_list, f3_list, name2, name3)
    print(len(merge_list))


    with open('./data/' + pair_name1 + '_and_' + pair_name2 +'.csv', 'w') as f:
        fnames = ['goal', 'dialog_1', 'dialog_2', 'order', 'goal_id']
        writer = csv.DictWriter(f, fieldnames=fnames) 
        writer.writeheader()
        writer.writerows(merge_list)

def rewrite_goal(goal):
    # goal_list = read_json('./data/goal.json')
    # goal_list = goal_list[:10]
    word2slot = {}
    for k, v in slot2word.items():
        word2slot[v] = k
    multiwoz_template_nlg = MultiwozTemplateNLG(True)
    # for goal in goal_list:
    goal_new = {}
    reqt_information = ' '
    for domain_k, domain_v in goal.items():
        goal_one_domian = {}
        sent_reqt=''
        if domain_k!='domain_ordering':
            for act_k, act_v in domain_v.items():
                if act_k in ['info', 'book']:
                    # act_k = 'inform' if act_k=='info' else act_k
                    domain = domain_k.capitalize() + '-' + act_k.capitalize()
                    dia_list=[]
                    for k, v in act_v.items():
                        if k in word2slot:
                            dia_list.append([word2slot[k], v])
                        else:
                            dia_list.append([k, v])
                    goal_one_domian[domain]=dia_list

                if act_k == 'reqt':
                    sent_reqt = ' Looking for' + ' @' + domain_k + ' information: ' + ', '.join(act_v) + '. '
        # print(goal_one_domian)
            sentence = multiwoz_template_nlg.generate(goal_one_domian)
            reqt_information = reqt_information + ' <br>#Constraints about domain @'+domain_k + ": " + sentence + sent_reqt
        # print(sentence)
    return reqt_information








if __name__ == '__main__':
    # input_file = sys.argv[1]
    # output_file = sys.argv[2]
    # example(input_file, output_file)
    # multi-dense
    # example('model_wgan/2020-04-20-14-49-37/collected_dialog.json', 'model_wgan/2020-04-20-14-49-37/collected_dialog.txt')
    # example('model_wgan/2020-04-20-14-49-37/collected_dialog.json', 'test.txt')
    # adv
    # example('model_wgan/2020-04-20-14-55-55/collected_dialog.json', 'model_wgan/2020-04-20-14-55-55/collected_dialog.txt')
    # # dia_seq
    # example('model_seq/2020-04-20-15-01-55/collected_dialog.json', 'model_seq/2020-04-20-15-01-55/collected_dialog.txt')
    # # gdpl
    # example('../GDPL_emnlp/model_emnlp/2020-04-20-14-49-40/collected_dialog.json', '../GDPL_emnlp/model_emnlp/2020-04-20-14-49-40/collected_dialog.txt')


    # # multi-dense
    # 2020-04-19-13-45-27: old_tracker, 2020-04-20-14-49-37: new tracker
    # example_json('model_wgan/2020-04-20-14-49-37/collected_dialog.json', 'model_wgan/2020-04-20-14-49-37/collected_dialog_with_nlg.json')
    # # adv
    # 2020-04-19-13-55-05: old_tracker, 2020-04-20-14-55-55: new tracker
    example_json('model_wgan/2020-04-19-13-55-05/collected_dialog.json', 'model_wgan/2020-04-19-13-55-05/collected_dialog_with_nlg.json')
    # # dia_seq
    # example_json('model_seq/2020-04-20-15-01-55/collected_dialog.json', 'model_seq/2020-04-20-15-01-55/collected_dialog_with_nlg.json')
    # # gdpl
    example_json('../GDPL_emnlp/model_emnlp/2020-04-19-13-45-29/collected_dialog.json', '../GDPL_emnlp/model_emnlp/2020-04-19-13-45-29/collected_dialog_with_nlg.json')


    # build_crowdsource_pair('mdense', 'model_wgan/2020-04-20-14-49-37/collected_dialog_with_nlg.json', 'gdpl', '../GDPL_emnlp/model_emnlp/2020-04-20-14-49-40/collected_dialog_with_nlg.json')
    # build_crowdsource_pair('adv', 'model_wgan/2020-04-20-14-55-55/collected_dialog_with_nlg.json', 'gdpl', '../GDPL_emnlp/model_emnlp/2020-04-20-14-49-40/collected_dialog_with_nlg.json', 100)
    # build_crowdsource_pair('seq', 'model_seq/2020-04-20-15-01-55/collected_dialog_with_nlg.json', 'gdpl', '../GDPL_emnlp/model_emnlp/2020-04-20-14-49-40/collected_dialog_with_nlg.json')

    build_crowdsource_pair('adv', 'model_wgan/2020-04-19-13-55-05/collected_dialog_with_nlg.json', 'gdpl', '../GDPL_emnlp/model_emnlp/2020-04-19-13-45-29/collected_dialog_with_nlg.json')
    
    # build_crowdsource_pair_for_all('mdense', 'model_wgan/2020-04-20-14-49-37/collected_dialog_with_nlg.json', 'seq', 'model_seq/2020-04-20-15-01-55/collected_dialog_with_nlg.json', 'gdpl', '../GDPL_emnlp/model_emnlp/2020-04-20-14-49-40/collected_dialog_with_nlg.json')


    # rewrite_goal()