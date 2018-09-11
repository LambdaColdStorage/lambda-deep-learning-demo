"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function


import importlib


class Inputter(object):
  def __init__(self, args):
    self.args = args
    self.augmenter = (None if not self.args.augmenter else
                      importlib.import_module(
                        "source.augmenter." + args.augmenter))

  def get_num_samples(self, *argv):
    pass

  def parse_fn(self, mode, *argv):
    pass

  def input_fn(self, mode, *argv):
    pass


def build(args):
  return Inputter(args)
