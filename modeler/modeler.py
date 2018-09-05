"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""


class Modeler(object):
  def __init__(self, args):
    self.args = args


def build(args):
  return Modeler(args)
