from cv2 import cv2
import numpy as np


class ImageObjects:
    "A Wrapper for ImageObjects' Objects"

    def __init__(self, img, mask, num_classes):

        self.img = img
        self.mask = mask

        self.num_classes = num_classes
        self.contours_list = []

    def show_mask(self):
        "Displays Segmentation Mask"

        "This will be the default window for inference"
        cv2.imshow("Image Objects", self.mask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_objects(self):
        "Updates the list of contours of detected image objects"

        "Iterate over every class and find its corresponding image object as a list of contours"
        for Class in range(self.num_classes):
            mask_temp = cv2.inRange(self.mask, np.array([Class]), np.array([Class]))
            contours_temp, _ = cv2.findContours(
                mask_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            self.contours_list.append(contours_temp)

    def draw_objects(self):
        "Returns np.ndarray image bounded by detected image object's contours"

        "Create a null matrix that matches the size of output"
        img_shape = self.mask.shape
        mask_output = np.zeros(img_shape)

        for Class in range(self.num_classes):
            print(f"contours_list = {len(self.contours_list)}")
            print(f"num_classes = {self.num_classes}")
            cv2.drawContours(
                mask_output, self.contours_list[Class], -1, (128, 128, 128), 3
            )

        return mask_output

    def show_object_contours(self):
        "Displays Image Object's Contours"

        mask_contours = self.draw_objects()
        cv2.imshow("Image Objects", mask_contours)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def inference(self):
        self.detect_objects()
        self.draw_objects()
        self.show_object_contours()
