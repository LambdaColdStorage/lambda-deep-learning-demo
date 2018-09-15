"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function


class Callback(object):
  def __init__(self, config):
    self.config = config

  def before_run(self, *argv):
    pass

  def after_run(self, *argv):
    pass

  def before_step(self, *argv):
    pass

  def after_step(self, *argv):
    pass


def build(config):
  return Callback(config)
