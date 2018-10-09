"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from modeler import Modeler


class ObjectDetectionModeler(Modeler):
  def __init__(self, args, net):
    super(ObjectDetectionModeler, self).__init__(args, net)
    

  def get_dataset_info(self, inputter):
    self.num_samples = inputter.get_num_samples()
    self.anchors = inputter.get_anchors()
    self.anchors_map = inputter.get_anchors_map()

  def create_nonreplicated_fn(self):
    pass

  def create_graph_fn(self, input):
    pass

  def create_eval_metrics_fn(self, predictions, labels):
    pass

  def create_loss_fn(self, logits, labels):
    pass

  def model_fn(self, x):
    images = x[0]
    classes = x[1]
    boxes = x[2]
    is_crowd = x[3]

    # create training target

    return images

def build(args, network):
  return ObjectDetectionModeler(args, network)
