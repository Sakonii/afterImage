import numpy as np
from cv2 import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class Segmentation:
    "A Wrapper for Segmentation related objects"

    def __init__(self, modelWeights, cfgPath):
        # Load Detection Model
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(cfgPath))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set detection threshold
        self.cfg.MODEL.WEIGHTS = modelWeights
        self.cfg.MODEL.DEVICE = "cpu"
        self.predict = DefaultPredictor(self.cfg)

    def update_threshold(self, threshValue):
        " Updates Detection Threshold On Trackbar Event"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshValue / 100
        self.predict = DefaultPredictor(self.cfg)

    def tensor_to_np(self):
        "Convert Tensors to OpenCV Comaptible Types"
        # Convert to np.ndarray
        self.mask = self.outputs["panoptic_seg"][0].numpy().astype(np.uint8)
        self.maskDetails = self.outputs["panoptic_seg"][1]

    def show(self):
        "Displays content input image and its corresponding detected objects"
        cv2.imshow("afterImage", self.img_display)
        #cv2.imshow("mask", self.mask)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def draw_segmentation(self, inPlace=False):
        "Draw content input image and its corresponding detected torch objects [not inplace]"
        v = Visualizer(
            self.img_display[:, :, ::-1],
            MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            scale=1.0,
        )
        v = v.draw_panoptic_seg_predictions(
            self.outputs["panoptic_seg"][0].to("cpu"), self.outputs["panoptic_seg"][1]
        )
        segImage = np.transpose(v.get_image()[:, :, ::-1], (0, 1, 2))
        if inPlace:
            self.img_display = segImage
        return segImage

    def start_segmentation(self, img):
        "Updates and Returns bboxes, preds, scores and classes for next video frame"
        self.img = img
        self.img_display = np.copy(self.img)
        # Model Prediction
        self.outputs = self.predict(img)
        self.tensor_to_np()
        # Display Segmentation
        self.draw_segmentation(inPlace=True)
        self.show()
        return (self.img, self.mask, self.maskDetails)
