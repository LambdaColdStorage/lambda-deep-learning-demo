"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function


class Callback(object):
  def __init__(self, args):
    self.args = args

  def before_run(self, *argv):
    pass

  def after_run(self, *argv):
    pass

  def before_step(self, *argv):
    pass

  def after_step(self, *argv):
    pass


def build(args):
  return Callback(args)
