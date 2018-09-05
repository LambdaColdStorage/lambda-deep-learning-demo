"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""


class Runner(object):
  def __init__(self, args, inputter, modeler):
    self.args = args
    self.inputter = inputter
    self.modeler = modeler


def build(args, inputter, modeler):
  return Runner(args, inputter, modeler)
