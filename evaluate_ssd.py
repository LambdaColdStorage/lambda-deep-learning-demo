import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DATASET_DIR = "/mnt/data/data/mscoco"
DATASET_META = "minival2014"

DETECTION_FILE = "/home/ubuntu/data/mscoco/results/SSD_300x300_score/detections_minival_ssd300_results.json"

annotation_file = os.path.join(
DATASET_DIR,
"annotations",
"instances_" + DATASET_META + ".json")
coco = COCO(annotation_file)
coco_results = coco.loadRes(DETECTION_FILE)

cocoEval = COCOeval(coco, coco_results, "bbox")

imgIds=sorted(coco.getImgIds())
imgIds=imgIds[0:4000]
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()