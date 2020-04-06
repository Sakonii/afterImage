from cv2 import cv2
import numpy as np

from inpainting import Inpainting
from segmentation import Segmentation


class UI:
    "A Wrapper for In-Window Interface Variables"

    def __init__(self):
        # Current display image and most recently pressed button
        self.imgDisplay = None
        self.pressedKey = None
        self.selectedObj = []

    def print_console(self):
        "Print on Console"
        print(
            "Usage:\n\t1.) Register Left-click to select an image object"
            "\n\t2.) Press 'd' to delete the object."
            "\n\t3.) Press 'f' to apply inpainting"
            "\n\t4.) Press 'r' to reset changes"
            "\n\t5.) Press 'q' to quit"
        )


class Inference:
    "A Wrapper for ImageObjects' Objects"

    def __init__(
        self, img, mask, maskDetails, inpaintModel, pathToLearner="./models",
    ):
        self.imgDisk = self.img = img
        self.mask = mask
        # Segmentation prediction details
        self.numOfObjects = len(maskDetails)
        self.maskDetails = maskDetails
        # Instantiate UI Components and Models
        self.ui = UI()
        self.contoursList = []
        # Inpaint model
        self.inpaintModel = inpaintModel
        self.pathToLearner = pathToLearner
        # This will be the default window for inference
        cv2.namedWindow(winname="afterImage")
        self.boolShowSeg = cv2.createTrackbar(
            "Show Segments", "afterImage", 0, 1, self._show_objects
        )
        cv2.setMouseCallback("afterImage", self.mouse_events)

    def contour_containing_point(self, x, y, boolDebug=True, bool=False):
        "Returns the contour under which the co-ordinate falls"
        for objectIndex in range(
            self.numOfObjects + 1
        ):  # for contours in self.contoursList:
            contours = self.contoursList[objectIndex]
            for contour in contours:
                if (
                    cv2.pointPolygonTest(contour=contour, pt=(x, y), measureDist=False)
                    >= 0
                    and contour is not self.ui.selectedObj
                ):
                    if boolDebug:
                        print(f"{self.maskDetails[objectIndex-1]}")
                    return True if bool else contour
                else:
                    pass
        return False if bool else []

    def show_mask(self):
        "Displays Segmentation Mask"
        cv2.imshow(winname="afterImage", mat=self.mask)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

    def detect_objects(self):
        "Updates the list of contours of detected image objects"
        # Iterate over every class and find its corresponding image object as a list of contours
        for objectIndex in range(self.numOfObjects + 1):
            maskTemp = cv2.inRange(
                src=self.mask,
                lowerb=np.array([objectIndex]),
                upperb=np.array([objectIndex]),
            )
            contours_temp, _ = cv2.findContours(
                image=maskTemp, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
            )
            self.contoursList.append(contours_temp)

    def draw_objects(self, boolShowSeg=False, onBlank=False, fromClean=False):
        "Returns np.ndarray image bounded by detected image object's contours"
        if onBlank:
            # Create a null matrix that matches the size of output
            self.ui.imgDisplay = np.zeros(shape=self.mask.shape)
        else:
            self.ui.imgDisplay = np.copy(self.img if fromClean else self.ui.imgDisplay)
        if boolShowSeg:
            for objectIndex in range(self.numOfObjects):
                cv2.drawContours(
                    image=self.ui.imgDisplay,
                    contours=self.contoursList[objectIndex],
                    contourIdx=-1,
                    color=(0, 200, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
        # Draws detected objects onto ui.imgDisplay
        cv2.drawContours(
            image=self.ui.imgDisplay,
            contours=self.ui.selectedObj,
            contourIdx=-1,
            color=(0, 215, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        return self.ui.imgDisplay

    def show_objects(self, fromClean=False):
        "Displays Image Objects"
        self.draw_objects(self.boolShowSeg, fromClean=fromClean)
        cv2.imshow(winname="afterImage", mat=self.ui.imgDisplay)

    def _show_objects(self, x=None):
        "Function Wrapper to switch Segmentation view (only for trackbar)"
        self.boolShowSeg = not self.boolShowSeg
        self.draw_objects(self.boolShowSeg, fromClean=True)
        cv2.imshow(winname="afterImage", mat=self.ui.imgDisplay)

    def obj_delete(self, inPlace=True):
        "Delete the selected object"
        self.ui.imgDisplay = np.copy(self.img)
        cv2.drawContours(
            image=self.ui.imgDisplay,
            contours=[self.ui.selectedObj],
            contourIdx=-1,
            color=(255, 255, 255),
            thickness=cv2.FILLED,
        )
        if inPlace:
            self.img_save(toDisk=False)

    def img_save(self, toDisk=True):
        "Save display image 'self.ui.imgDisplay' to actual image 'self.img' "
        self.ui.selectedObj = []
        self.img = self.ui.imgDisplay
        if toDisk:
            self.imgDisplay = self.ui.imgDisplay
            cv2.imwrite("./img_output/img.jpg", self.imgDisplay)

    def img_reload(self):
        "Re-loads display image from actual image"
        self.img = np.copy(self.imgDisplay)
        self.ui.imgDisplay = np.copy(self.imgDisplay)
        # Also Clear Selected Object
        self.ui.selectedObj = []
        self.show_objects(fromClean=False)

    def inpaint(self, inPlace=True):
        "Inpainting"
        self.ui.imgDisplay = Inpainting(self.img, self.inpaintModel).start_inpainting()
        if inPlace:
            self.img_save(toDisk=False)

    def keyboard_events(self):
        "Non-callback keyboard input function for image manipulation"

        while True:
            self.ui.pressedKey = cv2.waitKey(0)

            if self.ui.pressedKey & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return

            elif self.ui.pressedKey & 0xFF == ord("r"):
                self.img_reload()
                continue

            elif self.ui.pressedKey & 0xFF == ord("f"):
                print("Inpainting, this may take a while ...")
                self.inpaint()
                self.show_objects(fromClean=False)
                print("inpainted")
                continue

            if self.ui.selectedObj != []:
                # If any of the object is selected by mouse event
                if self.ui.pressedKey & 0xFF == ord("d"):
                    self.obj_delete()
                    self.show_objects(fromClean=False)
                    print("deleted")

                elif self.ui.pressedKey & 0xFF == ord("m"):
                    print("moving")

                elif self.ui.pressedKey & 0xFF == ord("s"):
                    print("Can't save. Un-select the object first.")

            elif self.ui.selectedObj == []:
                # If none of the object is selected
                if self.ui.pressedKey & 0xFF == ord("s"):
                    self.img_save()
                    print("Progress Saved!")
                else:
                    print(f"Trigger a defined keyboard or mouse event")

    def mouse_events(self, event, x, y, flags, param):
        "Mouse callback function definition"

        if event == cv2.EVENT_LBUTTONDOWN:
            self.ui.selectedObj = self.contour_containing_point(x, y)
            self.show_objects(fromClean=True)

        elif event == cv2.EVENT_MOUSEMOVE:
            "Later"

        elif event == cv2.EVENT_LBUTTONUP:
            "Later"

    def inference(self):
        "Run Image Manipulation inference"
        self.ui.print_console()
        self.detect_objects()
        self.show_objects(fromClean=True)
        self.keyboard_events()
