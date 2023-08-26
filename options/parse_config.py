# options.py ---

import argparse
import os


class HierarchyArgmentParser():
    def __init__(self, flatten_args=['experiment', 'train', 'eval', 'test']):
        super(HierarchyArgmentParser, self).__init__()
        self.flatten_args = flatten_args
        self.parser = argparse.ArgumentParser()
        self.sub = self.parser.add_subparsers()
        self.parser_list = {}

    def add_parser(self, name):
        args = self.sub.add_parser(name)
        self.parser_list[name] = args
        return args

    def parse_args(self):
        opt_all, _ = self.parser.parse_known_args()
        for name, parser in self.parser_list.items():
            opt, _ = parser.parse_known_args()
            if name in self.flatten_args:
                for key, value in vars(opt).items():
                    setattr(opt_all, key, value)
            else:
                setattr(opt_all, name, opt)
        return opt_all


def dump_args(opt):
    args = {}
    for k, v in vars(opt).items():
        if isinstance(v, argparse.Namespace):
            args[k] = vars(v)
        else:
            args[k] = v
    return args