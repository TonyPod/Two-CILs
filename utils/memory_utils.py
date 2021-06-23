# -*- coding:utf-8 -*-  

""" 
@time: 12/29/20 10:55 AM 
@author: Chen He 
@site:  
@file: memory_utils.py
@description:  
"""
from utils.episodic_memory import EpisodicMemory


def get_memory(args):
    if args.memory_type == 'episodic':
        return EpisodicMemory(args)
    elif args.memory_type == 'none':
        return None
    else:
        raise Exception('Invalid memory type')
