"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Inputter(object):
  def __init__(self, args):
    self.args = args

  @abc.abstractmethod
  def get_num_samples(self, *argv):
    raise NotImplementedError()

  @abc.abstractmethod
  def parse_fn(self, mode, *argv):
    raise NotImplementedError()

  @abc.abstractmethod
  def input_fn(self, mode, *argv):
    raise NotImplementedError()
 

def build(args):
  return Inputter(args)
