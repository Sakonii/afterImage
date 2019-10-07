  
import numpy as np
from cv2 import cv2

from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *

from cv2 import cv2


class Segmentation:  

    def __init__(self, fname_img):

        self.path_to_learner = './models'
        self.learn = load_learner(path = self.path_to_learner, file = '1_segmentation_model.pkl')

        self.img_path = './img_input/' + fname_img
        self.img  = open_image(self.img_path)
        

    global acc_camvid
    def acc_camvid(self, input, target):
        "Dummy accuracy metric (not required for inference)"
        return None  

    
    def start_segmentation(self):

        self.pred = self.learn.predict(self.img) 
        self.img.show(y = self.pred[0], figsize = (12,12))

        # Convert from torch style image to OpenCV compatible numpy array
        self.img = image2np(self.img.data * 255).astype(np.uint8)
        cv2.cvtColor(src = self.img, dst = self.img, code = cv2.COLOR_BGR2RGB)

        cv2.imshow('image', self.img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


