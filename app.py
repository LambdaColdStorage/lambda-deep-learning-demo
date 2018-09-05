"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import importlib

import tensorflow as tf

class APP(object):
  """ A machine leanring application composed of 
      an inputter, a modeler and a runner.
  """
  def __init__(self, args):
    self.args = args
    self.inputter = importlib.import_module("inputter." + args.inputter).build(self.args)
    self.modeler = importlib.import_module("modeler." + args.modeler).build(self.args)
    self.runner = importlib.import_module(
      "runner." + args.runner).build(self.args,
                                     self.inputter,
                                     self.modeler)


  def run(self):
    # print("Applicaiton is running.")
    self.runner.run()
