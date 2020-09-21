import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import scipy.stats as stats
matplotlib.rcParams.update({'font.size': 16})

"""
INFO:root:reward -1422.5165013517114
INFO:root:turn 7.092
INFO:root:match 0.8326817826426895
INFO:root:inform rec 0.8869182746878547, F1 0.8799887379460829
INFO:root:success 0.826   138577  slurm_logs/irl_mt_newparam_lambda_138117_
"""
def fetch_file_list():
    # file_pre = 'slurm_logs/irl_mt_forward_newparam_lambda_138160_'    # this is forward, clip=0.2, save_freq=1
    # file_pre = 'slurm_logs/irl_mt_forward_newparam_lambda_138577_'

    # file_pre = 'slurm_logs/irl_mt_opposite_newparam_lambda_138277_'    # this is backward, clip=0.2, save_freq=1
    # file_pre = 'slurm_logs/irl_mt_newparam_lambda_138117_'

    # file_pre = 'slurm_logs/irl_mt_opposite_prepolicy_clip0.03_lambda_139809_'    # this is backward, clip=0.03, save_freq=1, filename='model_saved_ori/model_agenda_pre_ori/best'
    # file_pre = 'slurm_logs/irl_mt_forward_prepolicy_clip0.03_lambda_139860_'    # this is forward, clip=0.03, save_freq=1, filename='model_saved_ori/model_agenda_pre_ori/best'

    file_pre = 'slurm_logs/irl_mt_opposite_prepolicy_2_clip0.03_lambda_140333_'    # this is backward, clip=0.03, save_freq=1, filename='model_saved_ori/model_saved/model_agenda_pre_mt_op_1.0/best'
    file_pre = 'slurm_logs/irl_mt_forward_prepolicy_2_clip0.03_lambda_140360_'    # this is forward, clip=0.03, save_freq=1, filename='model_saved_ori/model_saved/model_agenda_pre_mt_op_1.0/best'

    # file_pre = 'slurm_logs/irl_mt_opposite_prepolicy_2_clip0.02_lambda_140604_'    # this is backward, clip=0.03, save_freq=1, filename='model_saved_ori/model_saved/model_agenda_pre_mt_op_1.0/best'
    file_pre = 'slurm_logs/irl_mt_forward_prepolicy_2_clip0.02_lambda_140601_'    # this is forward, clip=0.03, save_freq=1, filename='model_saved_ori/model_saved/model_agenda_pre_mt_op_1.0/best'
    
    file_pre = 'slurm_logs/irl_mt_opposite_small_clip0.03_141965_'


    for i in range(1, 111, 5):
        turn, match, inform, f1, succ = [], [], [], [], []
        for j in range(i,i+5):
            file = file_pre + str(j) + '.log'
            with open(file, 'r') as f:
                f_h = f.readlines()[-4:]
                float(f_h[0].strip().split()[-1])
                turn.append(float(f_h[0].strip().split()[-1]))
                match.append(float(f_h[1].strip().split()[-1]))
                inform.append(float(f_h[2].strip().split()[2].replace(',','')))
                f1.append(float(f_h[2].strip().split()[4]))
                succ.append(float(f_h[3].strip().split()[-1]))
        num = i//5 if i<=55 else (i-55)//5
        print(i, num * 0.2)
        print(turn)
        print(match)
        print(inform)
        print(f1)
        print(succ)

        print(np.around(np.array(turn).mean(),3), np.around(np.array(match).mean(), 3),np.around(np.array(inform).mean(), 3),np.around(np.array(f1).mean(), 3),np.around(np.array(succ).mean(), 3),)

def fetch_file_list_emnlp():
    # file_pre = 'slurm_logs/irl_mt_forward_newparam_lambda_138160_'    # this is forward, clip=0.2, save_freq=1
    # file_pre = 'slurm_logs/irl_mt_forward_newparam_lambda_138577_'

    # file_pre = 'slurm_logs/irl_mt_opposite_newparam_lambda_138277_'    # this is backward, clip=0.2, save_freq=1
    # file_pre = 'slurm_logs/irl_mt_newparam_lambda_138117_'

    # file_pre = 'slurm_logs/irl_mt_opposite_prepolicy_clip0.03_lambda_139809_'    # this is backward, clip=0.03, save_freq=1, filename='model_saved_ori/model_agenda_pre_ori/best'
    # file_pre = 'slurm_logs/irl_mt_forward_prepolicy_clip0.03_lambda_139860_'    # this is forward, clip=0.03, save_freq=1, filename='model_saved_ori/model_agenda_pre_ori/best'

    file_pre = 'slurm_logs/irl_mt_opposite_prepolicy_2_clip0.03_lambda_140333_'    # this is backward, clip=0.03, save_freq=1, filename='model_saved_ori/model_saved/model_agenda_pre_mt_op_1.0/best'
    file_pre = 'slurm_logs/irl_mt_forward_prepolicy_2_clip0.03_lambda_140360_'    # this is forward, clip=0.03, save_freq=1, filename='model_saved_ori/model_saved/model_agenda_pre_mt_op_1.0/best'

    # file_pre = 'slurm_logs/irl_mt_opposite_prepolicy_2_clip0.02_lambda_140604_'    # this is backward, clip=0.03, save_freq=1, filename='model_saved_ori/model_saved/model_agenda_pre_mt_op_1.0/best'
    file_pre = 'slurm_logs/irl_mt_forward_prepolicy_2_clip0.02_lambda_140601_'    # this is forward, clip=0.03, save_freq=1, filename='model_saved_ori/model_saved/model_agenda_pre_mt_op_1.0/best'
    
    file_pre = '../GDPL_emnlp/slurm_logs/data_irl_ori_clip0.03_train16_16_oldtracker_192057_'
    file_pre = '../GDPL_emnlp/slurm_logs/diffpre_irl_ori_clip0.03_trainx_16_oldtracker_194408_'


    for i in range(21, 40, 5):
        turn, match, inform, f1, succ = [], [], [], [], []
        for j in range(i,i+5):
            file = file_pre + str(j) + '.log'
            with open(file, 'r') as f:
                f_h = f.readlines()[-4:]
                float(f_h[0].strip().split()[-1])
                turn.append(float(f_h[0].strip().split()[-1]))
                match.append(float(f_h[1].strip().split()[-1]))
                inform.append(float(f_h[2].strip().split()[2].replace(',','')))
                f1.append(float(f_h[2].strip().split()[4]))
                succ.append(float(f_h[3].strip().split()[-1]))
        # num = i//5 if i<=55 else (i-55)//5
        # print(i, num * 0.2)
        # print(turn)
        # print(match)
        # print(inform)
        # print(f1)
        # print(succ)

        print(np.around(np.array(turn).mean(),3), np.around(np.array(match).mean(), 3),np.around(np.array(inform).mean(), 3),np.around(np.array(f1).mean(), 3),np.around(np.array(succ).mean(), 3),)

def fetch_file_list_adv():

#     file_pre = './slurm_logs/data_adv_nosim_wgan_realtrain16_32_temp0.005_oldtracker_194285_'   # this is for different data ratio
    # file_pre = '../GDPL_emnlp/slurm_logs/data_irl_ori_clip0.03_train16_16_oldtracker_192057_'
    # file_pre = '../GDPL_emnlp/slurm_logs/diffpre_irl_ori_clip0.03_trainx_16_oldtracker_194408_'
    file_pre = './slurm_logs/diffpre_adv_nosim_wgan_realtrainx_32_temp0.005_oldtracker_194316_'
    for i in range(21, 40, 5):
        turn, match, inform, f1, succ = [], [], [], [], []
        for j in range(i,i+5):
            file = file_pre + str(j) + '.log'
            with open(file, 'r') as f:
                all_lines = f.readlines()
                all_lines = all_lines[::-1]
                for l_n, line in enumerate(all_lines):
                    if line.strip().split()[0]=='INFO:root:reward' and all_lines[l_n+1].strip().split()[0]!='INFO:root:model' and int(all_lines[l_n+1].strip().split()[1].replace(",",""))<16:
                    # if line.strip().split()[0]=='INFO:root:reward':
                        # f_h = all_lines[l_n+1:l_n+6]  
                        f_h = all_lines[l_n-5:l_n][::-1]  
                        # print(f_h)
                        break  
                # f_h = f.readlines()[-4:]
                turn.append(float(f_h[0].strip().split()[-1]))
                match.append(float(f_h[1].strip().split()[-1]))
                inform.append(float(f_h[2].strip().split()[2].replace(',','')))
                f1.append(float(f_h[2].strip().split()[4]))
                succ.append(float(f_h[3].strip().split()[-1]))
        # num = i//5 if i<=55 else (i-55)//5
        # print(i, num * 0.2)
        # print(turn)
        # print(match)
        # print(inform)
        # print(f1)
        # print(succ)

        print(np.around(np.array(turn).mean(),3), np.around(np.array(match).mean(), 3),np.around(np.array(inform).mean(), 3),np.around(np.array(f1).mean(), 3),np.around(np.array(succ).mean(), 3),)

def draw_fig():
    fig = plt.figure(figsize=(10,5))
    mk = ('4', '+', '.', '2', '|', 4, '1', 5, 6, 7)
    colors = ('#e58e26', '#b71540', '#0c2461', '#0a3d62', '#079992', '#fad390', '#6a89cc','#60a3bc', '#78e08f')
        
    X = [0.1, 0.4, 0.7, 1.0]
    adv=[37.2, 81.6, 87., 88.4]
    gdpl=[21.2, 68.0, 73.3, 81.7]
    mdense = [27, 79.4, 85.1, 86.6]
    mclass = [31.7, 59.0, 53.6, 57.2]

    plt.plot(X, adv, label='DiaAdv', marker=mk[0],\
            color=colors[0],  linewidth=3)
    plt.plot(X, gdpl, label='GDPL', marker=mk[1],\
            color=colors[1],  linewidth=3)
    plt.plot(X, mdense, label='DiaMultiDense', marker=mk[2],\
            color=colors[2],  linewidth=3)
    plt.plot(X, mclass, label='DiaMultiClass', marker=mk[3],\
            color=colors[3],  linewidth=3)
 
    plt.axis([0, 1.1, 20, 90])
    plt.yticks(np.arange(20.0,90,10.))
    # plt.xticks(np.arange(0,self.length*1000,self.length*1000//250 * 10))
    plt.xticks([0.0, 0.1, 0.4, 0.7, 1.0])
    # plt.xticks([1000] + np.arange(10000,self.length*1000, 5000).tolist())
    plt.xlabel("Data Size", fontsize=18)
    plt.ylabel("Success rate", fontsize=18)
    plt.grid(True,  linestyle='-.', linewidth=0.7)

    leg = plt.legend(loc=0, fancybox=True, fontsize=16)
    # leg.get_frame().set_alpha(0.5)
    # fig.savefig("datasize.png", bbox_inches='tight')
    fig.savefig("datasize.pdf", bbox_inches='tight')


def draw_fig_pretrain():
    fig = plt.figure(figsize=(10,5))
    mk = ('4', '+', '.', '2', '|', 4, '1', 5, 6, 7)
    colors = ('#e58e26', '#b71540', '#0c2461', '#0a3d62', '#079992', '#fad390', '#6a89cc','#60a3bc', '#78e08f')
        
    X = [1, 4, 7, 10]
    adv=[14.1, 77.3, 84.2, 86.5] #  replace this one 
    gdpl=[10.4, 48.3, 62.2, 73.2]
    mdense = [10.2, 72.2, 81.6, 84.8]
    mclass = [12.6, 55.9, 55.6, 54.6]

    plt.plot(X, adv, label='DiaAdv', marker=mk[0],\
            color=colors[0],  linewidth=3)
    plt.plot(X, gdpl, label='GDPL', marker=mk[1],\
            color=colors[1],  linewidth=3)
    plt.plot(X, mdense, label='DiaMultiDense', marker=mk[2],\
            color=colors[2],  linewidth=3)
    plt.plot(X, mclass, label='DiaMultiClass', marker=mk[3],\
            color=colors[3],  linewidth=3)
 
    plt.axis([0, 11, 0, 90])
    plt.yticks(np.arange(0.0,90,10.))
    # plt.xticks(np.arange(0,self.length*1000,self.length*1000//250 * 10))
    plt.xticks([0, 1, 4, 7, 10])
    # plt.xticks([1000] + np.arange(10000,self.length*1000, 5000).tolist())
    plt.xlabel("Pretrain Epoch", fontsize=18)
    plt.ylabel("Success rate", fontsize=18)
    plt.grid(True, linestyle='-.', linewidth=0.7)

    leg = plt.legend(loc=0, fancybox=True, fontsize=16)
    # leg.get_frame().set_alpha(0.5)
    # fig.savefig("datasize.png", bbox_inches='tight')
    fig.savefig("diffpretrain.pdf", bbox_inches='tight')


if __name__ == '__main__':
#     fetch_file_list()
    # INFO:root:reward
#     fetch_file_list_emnlp()
#     fetch_file_list_adv()
    draw_fig()
    draw_fig_pretrain()

