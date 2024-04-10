# Some basic setup:
# Setup detectron2 logger
import cv2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from configs.mrcnn_config import get_config
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetMapper

from mrcnn.hook import LossEvalHook

def get_predictor(ckpt_path, roi_threshold=0.7):
    dtrn_cfg = get_cfg()
    # model configuration
    dtrn_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # training configuration
    dtrn_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    dtrn_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    dtrn_cfg.INPUT.MASK_FORMAT = "bitmask"
    dtrn_cfg.MODEL.WEIGHTS = ckpt_path
    dtrn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_threshold
    predictor = DefaultPredictor(dtrn_cfg)
    return predictor

def get_dtrn_cfg(cfg):
    dtrn_cfg = get_cfg()
    # model configuration
    dtrn_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    dtrn_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # training configuration
    dtrn_cfg.DATASETS.TRAIN = ("ToolEE/train",)
    dtrn_cfg.DATASETS.TEST = ("ToolEE/val", "ToolEE/novel")
    dtrn_cfg.DATALOADER.NUM_WORKERS = cfg.num_workers

    dtrn_cfg.SOLVER.IMS_PER_BATCH = cfg.batch_size
    dtrn_cfg.SOLVER.BASE_LR = cfg.lr  # pick a good LR
    dtrn_cfg.SOLVER.MAX_ITER = cfg.max_iter
    dtrn_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    dtrn_cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg.num_classes
    dtrn_cfg.OUTPUT_DIR = os.path.join(os.path.dirname(cfg.data_root), "mrcnn_result", "output")
    dtrn_cfg.INPUT.MASK_FORMAT = "bitmask"
    dtrn_cfg.TEST.EVAL_PERIOD = cfg.eval_freq
    return dtrn_cfg

class ToolEETainer(DefaultTrainer):
    def __init__(self, dtrn_cfg):
        self.dtrn_cfg = dtrn_cfg
        super(ToolEETainer, self).__init__(cfg=dtrn_cfg)
        self.resume_or_load(resume=False)
        
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.dtrn_cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.dtrn_cfg,
                self.dtrn_cfg.DATASETS.TEST[0],
                DatasetMapper(self.dtrn_cfg, True)
            )
        ))
        return hooks
    
    @classmethod
    def build_evaluator(cls, dtrn_cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(dtrn_cfg.OUTPUT_DIR, f"val{dataset_name}")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

def get_trainer(cfg):
    dtrn_cfg = get_dtrn_cfg(cfg)
    os.makedirs(dtrn_cfg.OUTPUT_DIR, exist_ok=True)
    trainer = ToolEETainer(dtrn_cfg)
    return trainer

if __name__ == '__main__':
    cfg = get_config()
    get_trainer(cfg)