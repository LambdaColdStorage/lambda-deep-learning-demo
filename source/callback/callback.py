"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Callback(object):
  def __init__(self, args):
    self.args = args

  @abc.abstractmethod
  def before_run(self, *argv):
    raise NotImplementedError()

  @abc.abstractmethod
  def after_run(self, *argv):
    raise NotImplementedError()

  @abc.abstractmethod
  def before_step(self, *argv):
    raise NotImplementedError()

  @abc.abstractmethod
  def after_step(self, *argv):
    raise NotImplementedError()


def build(args):
  return Callback(args)
