 
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *

from cv2 import cv2


class inference:  
    
    def start_inference(self):

        camvid = untar_data(URLs.CAMVID_TINY)

        path_lbl = camvid/'labels'
        path_img = camvid/'images'

        codes = np.loadtxt(camvid/'codes.txt', dtype = str)
        get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'

        data = (SegmentationItemList.from_folder(path_img)
                .split_by_rand_pct()
                .label_from_func(get_y_fn, classes = codes)
                .transform(get_transforms(), tfm_y = True, size = 360)
                .databunch(bs = 3, path = camvid)
                .normalize(imagenet_stats))

        learn = unet_learner(data, models.resnet34).load('1_segmentation_final.pth')
        learn.export()

        img = cv2.imread("img1.jpg")
        learn = load_learner(camvid)
        learn.predict(img)
