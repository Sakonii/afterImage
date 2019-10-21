import numpy as np
from cv2 import cv2

from fastai.vision import open_image, image2np
from fastai.callbacks.hooks import load_learner
from fastai.utils.mem import gc


class Segmentation:
    "A Wrapper for Segmentation related objects"

    def __init__(self, fname_img):

        self.path_to_learner = "./models"
        self.learn = load_learner(
            path=self.path_to_learner, file="1_segmentation_model.pkl"
        )

        "learn.data -> DataBunch, learn.data.c -> count(learn.data.classes)"
        self.num_classes = self.learn.data.c

        self.img_path = "./img_input/" + fname_img
        self.img = open_image(self.img_path)

    def show(self):
        "Displays content input image and its corresponding segmentation in separate windows"

        cv2.imshow("image", self.img)
        cv2.imshow("mask", self.mask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    global acc_camvid

    def acc_camvid(self, input, target):
        "Dummy accuracy metric (metric not required for inference)"
        return None

    def start_segmentation(self):
        "Returns np.ndarray content input image, its segmented mask and the number of classes in the model"

        self.pred = self.learn.predict(self.img)
        self.mask = self.pred[0]

        del self.pred
        gc.collect()

        "Convert from torch style Image to OpenCV compatible np.ndarray"
        self.img = image2np(self.img.data * 255).astype(np.uint8)
        cv2.cvtColor(src=self.img, dst=self.img, code=cv2.COLOR_BGR2RGB)

        self.mask = image2np(self.mask.data).astype(np.uint8)
        cv2.cvtColor(src=self.mask, dst=self.mask, code=cv2.COLOR_BGR2RGB)

        return self.img, self.mask, self.num_classes
