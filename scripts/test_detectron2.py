import torch
import detectron2

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available?", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))

print("Detectron2 version:", detectron2.__version__)

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)

cfg.MODEL.DEVICE = "cuda"

predictor = DefaultPredictor(cfg)

print("Detectron2 predictor constructed successfully!")