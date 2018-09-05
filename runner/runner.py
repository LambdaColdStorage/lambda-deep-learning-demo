"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""


class Runner(object):
  def __init__(self, args):
    self.args = args


def build(args):
  return Runner(args)
