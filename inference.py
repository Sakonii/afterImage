from cv2 import cv2
import numpy as np

from inpainting.inference import Inpainting
from segmentation import Segmentation


class UI:
    "A Wrapper for In-Window Interface Variables"

    def __init__(self):
        # Current display image / most recently pressed button / contours
        self.imgDisplay = None
        self.imgBuffer = None
        self.pressedKey = None
        self.selectedObj = []
        self.deletedObjContours = []
        self.moving = False
        self.copying = False

    def print_console(self):
        "Print on Console"
        print(
            "Usage:\n\t1.) Register Left-click to select an image object"
            "\n\t2.) Press 'd' to delete the object"
            "\n\t3.) Press 'm' to move the object"
            "\n\t4.) Press 'f' to apply fix"
            "\n\t5.) Press 'r' to reset changes"
            "\n\t6.) Press 'q' to quit"
        )

    def selectedObj_to_mask(self, maskSize):
        "Returns mask of currently selected object"
        mask = np.zeros(shape=maskSize, dtype=np.uint8)
        cv2.drawContours(
            image=mask,
            contours=[self.selectedObj],
            contourIdx=-1,
            color=(255,255,255),
            thickness=cv2.FILLED,
        )
        return mask


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

    def obj_delete(self, preserveImg=False, inPlace=True):
        "Delete the selected object"
        if not preserveImg:
            self.ui.imgDisplay = np.copy(self.img)
        # Create mask for inpainting
        inpaintMask = self.ui.selectedObj_to_mask(maskSize=self.mask.shape)
        # Inpaint in the mask
        self.ui.imgDisplay = cv2.inpaint(
            self.ui.imgDisplay, inpaintMask, 2, cv2.INPAINT_TELEA
        )
        # Append this object to list of deleted objects
        self.ui.deletedObjContours.append(self.ui.selectedObj)
        if inPlace:
            self.img_save(toDisk=False)

    def img_save(self, toDisk=True, unselectObj=True):
        "Save display image 'self.ui.imgDisplay' to actual image 'self.img' "
        self.ui.selectedObj = [] if unselectObj else self.ui.selectedObj
        self.img = self.ui.imgDisplay
        if toDisk:
            self.imgDisplay = self.ui.imgDisplay
            cv2.imwrite("./img_output/img.jpg", self.imgDisplay)

    def img_reload(self):
        "Re-loads display image from actual image"
        self.ui.imgDisplay = np.copy(self.img)
        # Also Clear Selected Object
        self.ui.selectedObj = []
        self.show_objects(fromClean=False)

    def inpaint(self, inPlace=True):
        "Inpainting"
        # Create mask for inpainting
        inpaintMask = self.ui.selectedObj_to_mask(maskSize=self.img.shape)
        self.ui.imgBuffer = np.copy(self.img)
        # Inpaint
        self.ui.imgDisplay = Inpainting(modelPath=self.inpaintModel).start_inpainting(
            self.ui.imgBuffer, inpaintMask
        )
        if inPlace:
            self.img_save(toDisk=False)

    def move_during(self, mouse_x, mouse_y):
        "Move the selected object"
        self.ui.imgBuffer = self.ui.imgDisplay = np.copy(self.img)
        # Create mask for object movement (src)
        moveMask = self.ui.selectedObj_to_mask(self.mask.shape)
        # Crop to ROI of selected object (minAreaRect) (src)
        x, y, w, h = cv2.boundingRect(self.ui.selectedObj)
        roiSrc = self.ui.imgBuffer[y : y + h, x : x + w]
        roiMask = moveMask[y : y + h, x : x + w]
        # ROI of object destination (dst)
        roiDst = self.ui.imgBuffer[mouse_y : mouse_y + h, mouse_x : mouse_x + w]
        # Add contents of foreground to background based on mask
        maskInverted = cv2.bitwise_not(roiMask)
        imgBg = cv2.bitwise_and(roiSrc, roiSrc, mask=roiMask)
        imgFg = cv2.bitwise_and(roiDst, roiDst, mask=maskInverted)
        dst = cv2.add(imgBg, imgFg)
        self.ui.imgDisplay[mouse_y : mouse_y + h, mouse_x : mouse_x + w] = dst
        self.show_objects()
        # Store it to buffer (pass this buffer to copy_after function)
        self.ui.imgBuffer[mouse_y : mouse_y + h, mouse_x : mouse_x + w] = dst

    def move_after(self, inPlace=True):
        "Place the selected object after moving"
        self.ui.imgDisplay = np.copy(self.ui.imgBuffer)
        if self.ui.moving:
            self.obj_delete(preserveImg=True, inPlace=inPlace)
        if inPlace:
            self.img_save(toDisk=False, unselectObj=False)

        self.ui.moving = False

    def keyboard_events(self):
        "Non-callback keyboard input function for image manipulation"
        while True:
            self.ui.pressedKey = cv2.waitKey(0)
            # Q for QUIT
            if self.ui.pressedKey & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return
            # R for RESET
            elif self.ui.pressedKey & 0xFF == ord("r"):
                self.img_reload()
                continue
            # F for FIX
            elif self.ui.pressedKey & 0xFF == ord("f"):
                print("Inpainting, this may take a while ...")
                self.inpaint()
                self.show_objects(fromClean=False)
                print("inpainted")
                continue
            # If any of the object is selected by mouse event
            if self.ui.selectedObj != []:
                # D to DELETE
                if self.ui.pressedKey & 0xFF == ord("d"):
                    self.obj_delete()
                    self.show_objects(fromClean=False)
                    print("deleted")
                # M to MOVE, this boolean triggers mouse event for moving
                elif self.ui.pressedKey & 0xFF == ord("m"):
                    self.ui.moving = not self.ui.moving
                    if not self.ui.moving:
                        self.show_objects(fromClean=True)
                # C to COPY, this boolean triggers mouse event for copying
                elif self.ui.pressedKey & 0xFF == ord("c"):
                    self.ui.copying = not self.ui.copying
                    if not self.ui.copying:
                        self.show_objects(fromClean=True)
                # S to Save
                elif self.ui.pressedKey & 0xFF == ord("s"):
                    print("Can't save. Un-select the object first.")
            # If none of the object is selected
            elif self.ui.selectedObj == []:
                if self.ui.pressedKey & 0xFF == ord("s"):
                    self.img_save()
                    print("Progress Saved!")
                else:
                    print(f"Trigger a defined keyboard or mouse event")

    def mouse_events(self, event, x, y, flags, param):
        "Mouse callback function definition"
        # If not in move mode:
        if not (self.ui.moving or self.ui.copying):
            if event == cv2.EVENT_LBUTTONDOWN:
                "Select obj with left-click"
                self.ui.selectedObj = self.contour_containing_point(x, y)
                self.show_objects(fromClean=True)
        else:
            if event == cv2.EVENT_MOUSEMOVE:
                "Specify location for object placement"
                self.move_during(x, y)

            elif event == cv2.EVENT_LBUTTONUP:
                "Drop / place selected object"
                self.move_after()
                self.show_objects(fromClean=True)

    def inference(self):
        "Run Image Manipulation inference"
        self.ui.print_console()
        self.detect_objects()
        self.show_objects(fromClean=True)
        self.keyboard_events()
