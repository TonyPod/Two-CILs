# -*- coding:utf-8 -*-  

""" 
@time: 12/15/20 11:21 PM 
@author: Chen He 
@site:  
@file: stats.py
@description:  
"""

import os
import pickle

import numpy as np
import wandb


class ModelStat:
    max_avg_keys = ['internal_memory', 'model_size', 'rehearsal_memory_size']
    sum_keys = ['time']
    force_update_keys = ['model_size', 'rehearsal_memory_size']

    def __init__(self, args):
        self.stat_file = os.path.join(args.OUTPUT_FOLDER, 'stats.pkl')
        self.args = args
        if os.path.exists(self.stat_file):
            self.statistics = pickle.load(open(self.stat_file, 'rb'))
        else:
            self.statistics = {key: {} for key in ModelStat.max_avg_keys + ModelStat.sum_keys}

    def load(self, stat_file):
        self.statistics = pickle.load(open(stat_file, 'rb'))

    def put(self, measure, group_idx, value):
        assert measure in self.statistics
        # keep the old version
        if not group_idx in self.statistics[measure] or measure in ModelStat.force_update_keys:
            self.statistics[measure][group_idx] = value

    def get(self, measure):
        return self.statistics[measure]

    def save(self):
        pickle.dump(self.statistics, open(self.stat_file, 'wb'))
        for key in self.statistics:
            measure = self.statistics[key]
            indices = sorted(measure.keys())
            with open(os.path.join(self.args.OUTPUT_FOLDER, 'stat_%s.txt' % key), 'w') as fout:
                for idx in indices:
                    fout.write(str(measure[idx]) + os.linesep)

    def upload_wandb(self):
        for key in ModelStat.max_avg_keys:
            measure = self.statistics[key]
            values = list(measure.values())
            wandb.log({'max_%s' % key: np.max(values), 'avg_%s' % key: np.mean(values)})
        for key in ModelStat.sum_keys:
            measure = self.statistics[key]
            values = list(measure.values())
            wandb.log({'total_%s' % key: np.sum(values)})
