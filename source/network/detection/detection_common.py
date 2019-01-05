import numpy as np
from pycocotools.mask import iou

def np_iou(A, B):
  def to_xywh(box):
    box = box.copy()
    box[:, 2] -= box[:, 0]
    box[:, 3] -= box[:, 1]
    return box

  ret = iou(
    to_xywh(A), to_xywh(B),
    np.zeros((len(B),), dtype=np.bool))
  return ret


