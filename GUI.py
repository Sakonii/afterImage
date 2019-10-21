from cv2 import cv2
import numpy as np


class ImageObjects:
    "A Wrapper for ImageObjects' objects"

    def __init__(self, img, mask, num_classes):

        self.img = img
        self.mask = mask

    def show(self):
        "Displays Segmentation mask bounded by detected image object's contours"

        cv2.imshow("Image Objects", self.mask_temp)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_objects(self):
        "Returns contours i.e. vector<vector> of detected image object of car (5)"

        "Test for class 5 i.e. bicyclist"
        self.class_no = 5

        self.mask_temp = cv2.inRange(
            self.mask, np.array([self.class_no]), np.array([self.class_no])
        )
        self.contours_temp, _ = cv2.findContours(
            self.mask_temp, 3, cv2.CHAIN_APPROX_NONE
        )

        cv2.drawContours(self.mask_temp, self.contours_temp, -1, (255, 255, 255), 3)