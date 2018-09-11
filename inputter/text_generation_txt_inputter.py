"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import os

import tensorflow as tf

from inputter import Inputter


class TextGenerationTXTInputter(Inputter):
  def __init__(self, args):
    super(TextGenerationTXTInputter, self).__init__(args)

  def create_nonreplicated_fn(self):
    pass

  def get_num_samples(self):
    pass

  def get_samples_fn(self):
    pass

  def parse_fn(self, *argv):
    """Parse a single input sample
    """
    pass

  def input_fn(self, test_samples=[]):
    pass


def build(args):
  return TextGenerationTXTInputter(args)
