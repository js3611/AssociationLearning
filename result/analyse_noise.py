__author__ = 'joschlemper'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os.path as path
import re


def analyse_noise(dir_name = 'Happy50', file_name = 'rbm_metric'):
    f = open('../data/remote/{}/{}.txt'.format(dir_name, file_name), 'r')
    lines = f.readlines()
    f.close()
    i = 0
    result = {'25_25': [], 'noise0.1_25_25': [],
              'noise0.3_25_25': [], 'noise0.5_25_25': [],#}
              'noise0.7_25_25': [], 'noise0.9_25_25': [],
              }

    if 'dbn' in file_name:
        for k in result.keys():
            result['[FT] {}'.format(k)] = []

    n_types = len(result)

    def get_numbers(x):
        return re.findall('\d+.\d+|\d+', x)

    while i < len(lines):
        for j in xrange(n_types):
            base = i + 7 * j
            print lines[base]
            name, epoch = lines[base].split(',')
            epoch = get_numbers(epoch)[0]
            happy_res = get_numbers(lines[base + 3])
            sadness_res = get_numbers(lines[base + 4])
            total_res = get_numbers(lines[base + 6])

            ## Store f1-measure for sadness
            # result[name].append(float(sadness_res[3]))
            # Store total f1-measure
            result[name].append(float(total_res[2]))

        i += (7 * n_types)

    fig, ax = plt.subplots()
    sorted_arr = {}
    for k in result:
        res = result[k]
        if '[FT]' in k:
            continue
        if res:
            noise = get_numbers(k)[0] if k != '25_25' else '0.0'
            noise_name = '{} noise'.format(noise)
            sorted_arr[noise_name] = res

    for k in sorted(sorted_arr):
        res = sorted_arr[k]
        if '[FT]' in k:
            continue
        if res:
            plt.plot(np.arange(1, 26), res, label=k)

    plt.legend(loc=4)
    # plt.title('{}, {}'.format(dir_name,file_name))

    majorLocator   = MultipleLocator(10)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator   = MultipleLocator(5)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)

    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    # ax.set_xticks()

    plt.xticks(np.arange(0, 25, 5), np.arange(0, 50, 10))

    # plt.ylim(0,1.05) 
    plt.show()


for dir_name in ['Sad50', 'Sad25', 'Sad10']:
    # for file_name in ['metric']:#,'dbn2_metric2','dbn3_metric2']:
    for file_name in ['dbn2_metric2','dbn3_metric2']:
        analyse_noise(dir_name=dir_name, file_name=file_name)
