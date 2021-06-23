# -*- coding:utf-8 -*-  

""" 
@time: 12/10/20 10:41 AM 
@author: Chen He 
@site:  
@file: memory.py
@description:  
"""


class Memory:
    def __init__(self, args):
        self.args = args

    def load_prev(self):
        raise NotImplementedError

    def save(self, **kwargs):
        raise NotImplementedError

    def size(self):
        raise NotImplementedError
