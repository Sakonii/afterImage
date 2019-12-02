from cv2 import cv2
import numpy as np


class ImageObjects:
    "A Wrapper for ImageObjects' Objects"

    def __init__(self, img, mask, num_classes):

        self.img = img
        self.mask = mask

        self.num_classes = num_classes
        self.contours_list = []

        self.img_display = None
        self.obj_selected = None

        # This will be the default window for inference
        cv2.namedWindow("Image Objects")
        self.boolSegment = cv2.createTrackbar(
            "Show Segments", "Image Objects", 0, 1, self._show_objects
        )
        cv2.setMouseCallback("Image Objects", self.mouse_events)

    def contour_containing_point(self, x, y):
        "Returns the contour under which the co-ordinate falls"

        for Class in range(self.num_classes):
            contours = self.contours_list[Class]

            for contour in contours:
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    return contour

        return None

    def mouse_events(self, event, x, y, flags, param):
        "Mouse callback function definition"

        if event == cv2.EVENT_LBUTTONDOWN:
            self.obj_selected = self.contour_containing_point(x, y)
            self.show_objects()

        elif event == cv2.EVENT_MOUSEMOVE:
            "Later"

        elif event == cv2.EVENT_LBUTTONUP:
            "Later"

    def show_mask(self):
        "Displays Segmentation Mask"

        cv2.imshow("Image Objects", self.mask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_objects(self):
        "Updates the list of contours of detected image objects"

        # Iterate over every class and find its corresponding image object as a list of contours
        for Class in range(self.num_classes):
            mask_temp = cv2.inRange(self.mask, np.array([Class]), np.array([Class]))
            contours_temp, _ = cv2.findContours(
                mask_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            self.contours_list.append(contours_temp)

    def draw_objects(self, showSegmentation=False, onBlank=False):
        "Returns np.ndarray image bounded by detected image object's contours"

        if onBlank:
            # Create a null matrix that matches the size of output
            img_shape = self.mask.shape
            self.img_display = np.zeros(img_shape)
        else:
            # Draws detected objects onto img_display
            self.img_display = np.copy(self.img)

        if showSegmentation:
            for Class in range(self.num_classes):
                cv2.drawContours(
                    self.img_display, self.contours_list[Class], -1, (75, 120, 75), 1
                )

        cv2.drawContours(self.img_display, self.obj_selected, -1, (0, 215, 255), 1)
        return self.img_display

    def show_objects(self):
        "Displays Image Objects"

        self.draw_objects(self.boolSegment)
        cv2.imshow("Image Objects", self.img_display)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _show_objects(self, x=None):
        "Function Wrapper to switch Segmentation view"

        self.boolSegment = not self.boolSegment
        self.draw_objects(self.boolSegment)
        cv2.imshow("Image Objects", self.img_display)

    def inference(self):
        "Run Image Manipulation inference"
        self.detect_objects()
        self.show_objects()