__author__ = 'joschlemper'

import matplotlib.pyplot as plt
import re
import numpy as np
from kanade_loader import emotion_dict, emotion_rev_dict
import matplotlib.patches as mpatches

colors = ['b','g','r','c','m','y','k','w']


def get_empty_emo_array():
    d = {}
    for key in emotion_dict:
        d[key] = []
    return d


def adbn_rename(name):
    d = re.findall("\d+", name)
    str = 'L[{}-{}-{}]-R[{}-{}-{}]-{}'.format(d[1], d[2], d[3], d[5], d[6], d[7], d[8])
    return str


def get_p(line):
    print line
    return float(re.findall("\d+.\d+", line)[0])


def plot_adbn_hidden_ns():
    # Read file
    f = open('Result/secure_result.txt')
    lines = f.readlines()
    length = len(lines)

    # Initialise
    i = 1
    stats = {}

    i = 0
    while i + 34 < length:
        rbm_name = lines[i]
        print rbm_name
        if adbn_rename(rbm_name) not in stats.keys():
            rbm_stats = {}
            rbm_stats['active_h'] = get_empty_emo_array()
            rbm_stats['v_noisy_active_h'] = get_empty_emo_array()
            rbm_stats['zero'] = get_empty_emo_array()
            rbm_stats['binomial0.1'] = get_empty_emo_array()
        else:
            rbm_stats = stats[adbn_rename(rbm_name)]
        for j in xrange(0, 4):
            s_base = i + 1 + 8 * j
            s_type = lines[s_base].strip()
            print s_type
            for k in xrange(1, 8):
                line = lines[s_base + k]
                emo, percentages = line.split(":")
                rbm_stats[s_type][emo] += map(float, percentages.split(',')[:-1])

        stats[adbn_rename(rbm_name)] = rbm_stats
        i += 34
    print stats
    keys = [
        'L[625-250-250]-R[625-250-250]-250',
        'L[625-500-500]-R[625-250-250]-250',
        'L[625-1000-1000]-R[625-250-250]-250',
        'L[625-250-250]-R[625-250-250]-500',
        'L[625-500-500]-R[625-250-250]-500',
        'L[625-1000-1000]-R[625-250-250]-500',
        'L[625-250-250]-R[625-250-250]-1000',
        'L[625-500-500]-R[625-250-250]-1000',
        'L[625-1000-1000]-R[625-250-250]-1000'
    ]
    # plot result
    for s_type in ['v_noisy_active_h']:
        for key in keys:
            k = 0
            for emo in ['happy']:  # , 'anger', 'sadness']:
                rbm = stats[key]
                vs = rbm[s_type][emo]
                # plt.subplot(131 + k)
                points = vs[(0)::(11)]
                plt.plot(points, label=key)
                x1, x2, y1, y2 = plt.axis()
                plt.axis((x1, x2, 0, 1))
                plt.legend()
                # plt.xticks(range(10), [5 + x * 5 for x in range(10)])
                # plt.xlabel('Epochs')
                # plt.ylabel('Proportion')
                # plt.title('Proportion vs Epochs')
                k += 1
                # for s in xrange(3):
                # plt.subplot(331 + k)
                #     points = vs[(s * 4)::(11)]
                #     plt.plot(points)
                #     x1, x2, y1, y2 = plt.axis()
                #     plt.axis((x1, x2, 0, 1))
                #     # plt.xticks(range(10), [5 + x * 5 for x in range(10)])
                #     # plt.xlabel('Epochs')
                #     # plt.ylabel('Proportion')
                #     # plt.title('Proportion vs Epochs')
                #     k+=1
    plt.show()


def adbn_rename(lr, name):
    d = re.findall("\d+", name)
    str = '{}_L[{}-{}-{}]-R[{}-{}-{}]-{}'.format(lr, d[1], d[2], d[3], d[5], d[6], d[7], d[8])
    return str


def plot_adbn_h_n_lr(proj_name='Result/remote/ExperimentADBN2.txt',max_hn=3, max_lr=3, title='small_lr'):
    # Read file
    f = open(proj_name)
    lines = f.readlines()
    length = len(lines)

    # Initialise
    stats = {}
    i = 1
    ctr_hn = 0
    ctr_lr = 0
    meta_data_len = 8


    while i + meta_data_len * max_hn * max_lr < length:
        rbm_name = lines[i]
        print rbm_name
        print ctr_hn
        print ctr_lr

        if adbn_rename(ctr_lr, rbm_name) not in stats.keys():
            rbm_stats = {}
            rbm_stats['active_h'] = get_empty_emo_array()
            rbm_stats['v_noisy_active_h'] = get_empty_emo_array()
            rbm_stats['zero'] = get_empty_emo_array()
            rbm_stats['binomial0.1'] = get_empty_emo_array()
        else:
            rbm_stats = stats[adbn_rename(ctr_lr, rbm_name)]
        for j in xrange(0, 4):
            s_base = i + 1 + meta_data_len * j
            s_type = lines[s_base].strip()
            print s_type
            for k in xrange(1, meta_data_len):
                line = lines[s_base + k]
                emo, percentages = line.split(":")
                rbm_stats[s_type][emo] += map(float, percentages.split(',')[:-1])

        stats[adbn_rename(ctr_lr, rbm_name)] = rbm_stats
        i += 34
        ctr_hn = (ctr_hn + 1) % max_hn  # 3 for ExperimentADBN2, 4 or ExperimentADBN3,
        if ctr_hn == 0:
            ctr_lr = (ctr_lr + 1) % max_lr  # 4 for ExperimentADBN2, 2 or ExperimentADBN3,

    print stats
    keys = []
    for k in sorted(stats.keys()):
        print k
        # if '-50' in k or '2_' in k:
        #     continue
        # if '0_' in k or '2_' in k:
        #     continue

        keys.append(k)
    print len(keys), keys

    # plot result
    for s_type in ['v_noisy_active_h']:
        c = 0
        for key in keys:
            k = -1
            for emo in ['happy', 'anger', 'sadness']:
                rbm = stats[key]
                vs = rbm[s_type][emo]
                # plt.subplot(131 + k)
                points = vs[(0)::(4)]
                std = np.std(points)
                k += 1
                c += 1

                if std > 0.15:
                    continue

                plt.plot(points, color=colors[k])
                # plt.plot(points, label=key, color=colors[k])
                # plt.xticks(range(10), [5 + x * 5 for x in range(10)])
                # plt.xlabel('Epochs')
                # plt.ylabel('Proportion')
                # plt.title('Proportion vs Epochs')
                # for s in xrange(3):
                # plt.subplot(331 + k)
                #     points = vs[(s * 4)::(11)]
                #     plt.plot(points)
                #     x1, x2, y1, y2 = plt.axis()
                #     plt.axis((x1, x2, 0, 1))
                #     # plt.title('Proportion vs Epochs')
                #     k+=1
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 1.2))

    plt.plot(np.repeat(0.8, 10), 'b--', label='happy')
    plt.plot(np.repeat(0.1, 10), 'g--', label='anger')
    plt.plot(np.repeat(0.1, 10), 'r--', label='sadness')
    plt.xticks(range(10), [5 + x * 5 for x in range(10)])
    plt.xlabel('Epochs')
    plt.ylabel('Proportion')
    plt.legend(loc=0)


    # plt.legend()
    # plt.title(title)
    plt.show()


def plot_dbn_hn(proj_name='Result/remote/ExperimentDBN.txt',title='Title'):
    # Read file
    f = open(proj_name)
    lines = f.readlines()
    length = len(lines)

    # Initialise
    i = 1
    stats = {}

    while i + 9 < length:
        rbm_name = lines[i]
        print rbm_name
        if rbm_name not in stats.keys():
            rbm_stats = get_empty_emo_array()
        else:
            rbm_stats = stats[rbm_name]
        s_base = i
        s_type = lines[s_base].strip()
        print s_type
        for k in xrange(1, 8):
            line = lines[s_base + k]
            emo, percentages = line.split(":")
            rbm_stats[emo] += map(float, percentages.split(',')[:-1])

        stats[rbm_name] = rbm_stats
        i += 9
    print stats
    keys = stats.keys()
    # plot result
    for key in keys:
        k = 0
        for emo in ['happy', 'anger', 'sadness']:
            rbm = stats[key]
            vs = rbm[emo]
            # plt.subplot(131 + k)
            points = vs[(0)::(4)]
            plt.plot(points, color=colors[k%3])
            # plt.plot(points, label=key, color=colors[k%3])
            x1, x2, y1, y2 = plt.axis()
            plt.axis((x1, x2, 0, 1.2))
            # plt.legend()
            # plt.xticks(range(10), [5 + x * 5 for x in range(10)])
            # plt.xlabel('Epochs')
            # plt.ylabel('Proportion')
            # plt.title('Proportion vs Epochs')
            k += 1
            # for s in xrange(3):
            # plt.subplot(331 + k)
            #     points = vs[(s * 4)::(11)]
            #     plt.plot(points)
            #     x1, x2, y1, y2 = plt.axis()
            #     plt.axis((x1, x2, 0, 1))
            #     # plt.xticks(range(10), [5 + x * 5 for x in range(10)])
            #     # plt.xlabel('Epochs')
            #     # plt.ylabel('Proportion')
            #     # plt.title('Proportion vs Epochs')
            #     k+=1
    plt.plot(np.repeat(0.8, 10), 'b--', label='happy')
    plt.plot(np.repeat(0.1, 10), 'g--', label='anger')
    plt.plot(np.repeat(0.1, 10), 'r--', label='sadness')
    plt.xticks(range(10), [5 + x * 5 for x in range(10)])
    plt.xlabel('Epochs')
    plt.ylabel('Proportion')
    plt.legend(loc=0)

    plt.show()

# plot_adbn_hidden_ns()
# plot_adbn_h_n_lr(proj_name='remote/ExperimentADBN2.txt',max_hn=3, max_lr=4,title='ADBN (low learning rate)')
# plot_adbn_h_n_lr(proj_name='remote/ExperimentADBN3.txt',max_hn=4, max_lr=2,title='ADBN (high learning rate)')
# plot_adbn_h_n_lr(proj_name='remote/ExperimentADBN4.txt',max_hn=3, max_lr=3,title='ADBN Small Architecture (dropout)')
# plot_adbn_h_n_lr(proj_name='remote/ExperimentADBN4long.txt',max_hn=3, max_lr=3,title='ADBN Small Architecture (dropout)')
# plot_adbn_h_n_lr(proj_name='ExperimentADBN4_stopped.txt',max_hn=3, max_lr=3,title='ADBN Small Architecture (dropout)')
# plot_adbn_h_n_lr(proj_name='remote/ExperimentADBN5.txt',max_hn=3, max_lr=3,title='ADBN Small Architecture (nodropout)')
# plot_adbn_h_n_lr(proj_name='remote/ExperimentADBN7.txt',max_hn=3, max_lr=3,title='ADBN Large Architecture (dropout)')
# plot_adbn_h_n_lr(proj_name='remote/ExperimentADBN6.txt',max_hn=3, max_lr=3,title='ADBN Large Architecture (nodropout)')
# plot_dbn_hn('remote/ExperimentDBN.txt',title='DBN h=500, lr=0.0001')
# plot_dbn_hn('remote/ExperimentDBN2.txt',title='DBN h=500,lr=0.001')
# plot_dbn_hn('remote/ExperimentDBN4.txt',title='DBN h=500,lr=0.01')
plot_dbn_hn('remote/ExperimentDBN3.txt',title='DBN h=250,lr=0.001')


def plot_result(file_name, mapping, architectures=['RBM', 'DBN', 'ADBN']):
    f = open(file_name, 'r')
    emotions = set(emotion_dict.keys())
    graphs = {'anger': [], 'happy': [], 'sadness': []}

    for line in f.readlines():
        splitted = ''.join(line.split()).split(':')
        if splitted[0] in graphs.keys():
            lab, val = splitted[0], splitted[1]
            graphs[lab].append(val)

    print map(len, graphs.values())
    print graphs
    f.close()

    # RBM
    anger = graphs['anger']
    happy = graphs['happy']
    sadness = graphs['sadness']
    child_reaction = mapping['sadness']
    plt.figure(1)
    len_arc = len(architectures)
    attempts = np.min(map(len, graphs.values())) / len_arc
    print attempts

    j = 0
    for i in [0,2,1]:
    # for i in xrange(0, len_arc):
        # plt.subplot(100 + len_arc * 10 + i + 1)
        plt.subplot(100 + len_arc * 10 + j + 1)
        plt.title(architectures[i])

        plt.plot(np.arange(0, attempts), np.repeat(child_reaction['anger'], attempts), 'r--')
        plt.plot(np.arange(0, attempts), np.repeat(child_reaction['sadness'], attempts), 'b--')
        plt.plot(np.arange(0, attempts), np.repeat(child_reaction['happy'], attempts), 'g--')

        plt.plot(np.arange(0, attempts), anger[i::len_arc], 'r', label='anger')
        plt.plot(np.arange(0, attempts), sadness[i::len_arc], 'b', label='sadness')
        plt.plot(np.arange(0, attempts), happy[i::len_arc], 'g', label='happy')
        plt.xticks(np.arange(0, 20, 2), np.arange(10, 220, 20))
        plt.ylim(0,1)
        j+=1
        # plt.legend()

    plt.legend(loc=0)
    # red_patch = mpatches.Patch(color='red', label='The red data')
    # plt.legend(handles=[red_patch])

    # plt.title(architectures[0])
    #
    # plt.plot(np.arange(0, attempts), np.repeat(child_reaction['anger'], attempts), 'r--')
    # plt.plot(np.arange(0, attempts), np.repeat(child_reaction['sadness'], attempts), 'b--')
    # plt.plot(np.arange(0, attempts), np.repeat(child_reaction['happy'], attempts), 'g--')
    #
    # plt.plot(np.arange(0, attempts), anger, 'r', label='anger')
    # plt.plot(np.arange(0, attempts), sadness, 'b', label='sadness')
    # plt.plot(np.arange(0, attempts), happy, 'g', label='happy')
    # plt.legend()

    plt.legend()

    plt.show()
    plt.savefig('.'.join([file_name.split('.')[0], 'png']))
    plt.close()



secure_mapping = ({'happy': {'happy': 0.8, 'anger': 0.1, 'sadness': 0.1},
                   'sadness': {'happy': 0.8, 'anger': 0.1, 'sadness': 0.1},
                   'anger': {'happy': 0.8, 'anger': 0.1, 'sadness': 0.1},
                   })

ambivalent_mapping = ({'happy': {'happy': 0.5, 'anger': 0.2, 'sadness': 0.3},
                           'anger': {'happy': 0.5, 'anger': 0.2, 'sadness': 0.3},
                           'sadness': {'happy': 0.5, 'anger': 0.2, 'sadness': 0.3},
                           })

avoidant_mapping = ({'happy': {'happy': 0.3, 'anger': 0.5, 'sadness': 0.2},
                     'anger': {'happy': 0.3, 'anger': 0.5, 'sadness': 0.2},
                     'sadness': {'happy': 0.3, 'anger': 0.5, 'sadness': 0.2},
                     })

# plot_result('../data/remote/kanade/Experiment3/Experiment3.txt', avoidant_mapping, architectures=['RBM', 'JDBN', 'ADBN'])
# plot_result('../data/remote/kanade/Experiment2/Experiment2.txt', ambivalent_mapping, architectures=['RBM', 'JDBN', 'ADBN'])
# plot_result('../data/remote/kanade/Experiment1_50/Experiment1_50.txt', secure_mapping, architectures=['RBM', 'JDBN', 'ADBN'])
# plot_result('../data/remote/Experiment7.txt', secure_mapping, architectures=['RBM', 'DBN', 'ADBN'])
# plot_result('../data/remote/Experiment7.txt', secure_mapping, architectures=['RBM', 'DBN', 'ADBN'])
# plot_result('../data/remote/Experiment7.txt', secure_mapping, architectures=['RBM', 'DBN', 'ADBN'])
# plot_result('data/remote/Experiment7_50.txt', secure_mapping, architectures=['DBN', 'ADBN'])
# plot_result('data/remote/Experiment8.txt', ambivalent_mapping, architectures=['DBN', 'ADBN'])
# plot_result('data/remote/Experiment8_50.txt', ambivalent_mapping, architectures=['DBN', 'ADBN'])
# plot_result('data/remote/Experiment9.txt', avoidant_mapping, architectures=['DBN', 'ADBN'])
# plot_result('data/remote/Experiment9_50.txt', avoidant_mapping, architectures=['DBN', 'ADBN'])

