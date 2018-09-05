"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from modeler import Modeler


class ImageClassificationModeler(Modeler):
  def __init__(self, args):
    super(ImageClassificationModeler, self).__init__(args)

  def create_precomputation(self):
    pass


def build(args):
  return ImageClassificationModeler(args)
