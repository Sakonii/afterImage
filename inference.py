from cv2 import cv2
import numpy as np

from inpainting import Inpainting


class Inference:
    "A Wrapper for ImageObjects' Objects"

    def __init__(
        self, img, mask, classes, model, path_to_learner="./models",
    ):
        self.img_disk = self.img = img
        self.mask = mask

        self.num_classes = len(classes)
        self.classes = classes

        self.img_display = None
        self.key_pressed = None

        self.obj_selected = []
        self.contours_list = []

        self.model = model
        self.path_to_learner = path_to_learner

        # This will be the default window for inference
        cv2.namedWindow(winname="Image Objects")
        self.bool_segment = cv2.createTrackbar(
            "Show Segments", "Image Objects", 0, 1, self._show_objects
        )
        cv2.setMouseCallback("Image Objects", self.mouse_events)

    def contour_containing_point(self, x, y, boolDebug=True):
        "Returns the contour under which the co-ordinate falls"

        for Class in range(self.num_classes):  # for contours in self.contours_list:
            contours = self.contours_list[Class]
            for contour in contours:
                if (
                    cv2.pointPolygonTest(contour=contour, pt=(x, y), measureDist=False)
                    >= 0
                    and contour is not self.obj_selected
                ):
                    if boolDebug:
                        print(f"{self.classes[Class]}")
                    return contour
                else:
                    pass
        return []

    def show_mask(self):
        "Displays Segmentation Mask"
        cv2.imshow(winname="Image Objects", mat=self.mask)

        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

    def detect_objects(self):
        "Updates the list of contours of detected image objects"

        # Iterate over every class and find its corresponding image object as a list of contours
        for _class in range(self.num_classes):
            mask_temp = cv2.inRange(
                src=self.mask, lowerb=np.array([_class]), upperb=np.array([_class])
            )
            contours_temp, _ = cv2.findContours(
                image=mask_temp, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
            )
            self.contours_list.append(contours_temp)

    def draw_objects(self, showSegmentation=False, onBlank=False, fromClean=False):
        "Returns np.ndarray image bounded by detected image object's contours"

        if onBlank:
            # Create a null matrix that matches the size of output
            img_shape = self.mask.shape
            self.img_display = np.zeros(shape=img_shape)
        else:
            self.img_display = np.copy(self.img if fromClean else self.img_display)

        if showSegmentation:
            for _class in range(self.num_classes):
                cv2.drawContours(
                    image=self.img_display,
                    contours=self.contours_list[_class],
                    contourIdx=-1,
                    color=(75, 120, 75),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

        # Draws detected objects onto img_display
        cv2.drawContours(
            image=self.img_display,
            contours=self.obj_selected,
            contourIdx=-1,
            color=(0, 215, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        return self.img_display

    def show_objects(self, fromClean=False):
        "Displays Image Objects"

        self.draw_objects(self.bool_segment, fromClean=fromClean)
        cv2.imshow(winname="Image Objects", mat=self.img_display)

    def _show_objects(self, x=None):
        "Function Wrapper to switch Segmentation view (only for trackbar)"

        self.bool_segment = not self.bool_segment
        self.draw_objects(self.bool_segment, fromClean=True)
        cv2.imshow(winname="Image Objects", mat=self.img_display)

    def obj_delete(self, inPlace=True):
        "Delete the selected object"
        self.img_display = np.copy(self.img)

        cv2.drawContours(
            image=self.img_display,
            contours=[self.obj_selected],
            contourIdx=-1,
            color=(255, 255, 255),
            thickness=cv2.FILLED,
        )
        if inPlace:
            self.img_save(toDisk=False)

    def img_save(self, toDisk=True):
        "Save display image 'self.img_display' to actual image 'self.img' "
        self.obj_selected = []
        self.img = self.img_display

        if toDisk:
            self.img_disk = self.img_display
            cv2.imwrite("./img_output/img.jpg", self.img_disk)

    def img_reload(self):
        "Re-loads display image from actual image"
        self.img = np.copy(self.img_disk)
        self.img_display = np.copy(self.img_disk)

        self.obj_selected = []
        self.show_objects(fromClean=False)

    def inpaint(self, inPlace=True):
        "Inpainting"
        self.img_display = Inpainting(self.img, self.model).start_inpainting()

        if inPlace:
            self.img_save(toDisk=False)

    def print_console(self):
        print(
            "Usage:\n\t1.) Register Left-click to select an image object"
            "\n\t2.) Press 'd' to delete the object."
            "\n\t3.) Press 'f' to apply inpainting"
            "\n\t4.) Press 'r' to reset changes"
            "\n\t5.) Press 'q' to quit"
        )

    def keyboard_events(self):
        "Non-callback keyboard input function for image manipulation"

        while True:
            self.key_pressed = cv2.waitKey(0)

            if self.key_pressed & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return

            elif self.key_pressed & 0xFF == ord("r"):
                self.img_reload()
                continue

            elif self.key_pressed & 0xFF == ord("f"):
                print("Inpainting, this may take a while ...")
                self.inpaint()
                self.show_objects(fromClean=False)
                print("inpainted")
                continue

            if self.obj_selected != []:
                # If any of the object is selected by mouse event
                if self.key_pressed & 0xFF == ord("d"):
                    self.obj_delete()
                    self.show_objects(fromClean=False)
                    print("deleted")

                elif self.key_pressed & 0xFF == ord("m"):
                    print("moving")

                elif self.key_pressed & 0xFF == ord("s"):
                    print("Can't save. Un-select the object first.")

            elif self.obj_selected == []:
                # If none of the object is selected
                if self.key_pressed & 0xFF == ord("s"):
                    self.img_save()
                    print("Progress Saved!")
                else:
                    print(f"Trigger a defined keyboard or mouse event")

    def mouse_events(self, event, x, y, flags, param):
        "Mouse callback function definition"

        if event == cv2.EVENT_LBUTTONDOWN:
            self.obj_selected = self.contour_containing_point(x, y)
            self.show_objects(fromClean=True)

        elif event == cv2.EVENT_MOUSEMOVE:
            "Later"

        elif event == cv2.EVENT_LBUTTONUP:
            "Later"

    def inference(self):
        "Run Image Manipulation inference"
        self.print_console()
        self.detect_objects()
        self.show_objects(fromClean=True)
        self.keyboard_events()
