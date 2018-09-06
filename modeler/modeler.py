"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Modeler(object):
  def __init__(self, args):
    self.args = args

  @abc.abstractmethod
  def create_precomputation(self, *argv):
    raise NotImplementedError()

  @abc.abstractmethod
  def model_fn(self, *argv):
    raise NotImplementedError()

  @abc.abstractmethod
  def create_graph_fn(self, *argv):
    raise NotImplementedError

  @abc.abstractmethod
  def create_eval_metrics_fn(self, *argv):
    raise NotImplementedError

  @abc.abstractmethod
  def create_loss_fn(self, *argv):
    raise NotImplementedError


def build(args):
  return Modeler(args)
